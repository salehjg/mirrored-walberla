//======================================================================================================================
//
//  This file is part of waLBerla. waLBerla is free software: you can
//  redistribute it and/or modify it under the terms of the GNU General Public
//  License as published by the Free Software Foundation, either version 3 of
//  the License, or (at your option) any later version.
//
//  waLBerla is distributed in the hope that it will be useful, but WITHOUT
//  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
//  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
//  for more details.
//
//  You should have received a copy of the GNU General Public License along
//  with waLBerla (see COPYING.txt). If not, see <http://www.gnu.org/licenses/>.
//
//! \file SweepgenLdcCPU.cpp
//! \brief Throughput (MLUP/s) of the SweepgenLbmCPU LBM sweeps in a lid-driven cavity, per code generation variant
//
//======================================================================================================================

//  The same app is built once per code generation variant (see CMakeLists.txt and SweepgenLdcCPU.py).
//  Each run prints one result line (README.md lists the fields), so the variants can be compared directly.
//
//  Usage: SweepgenLdcCPU_<variant> [cells per block and direction = 64] [timesteps = 200] [warmup = 10]
//
//  One block of cells^3 per MPI process; the blocks form the grid math::getFactors3D(processes). The
//  domain is not periodic. Walls as in waLBerla's UniformGridCPU LDC scenario: the north face (y max) is
//  a lid moving with (u_lid, 0, 0), the W, E, S, B and T faces are no-slip walls. A lattice link from
//  a fluid cell into a cell outside the domain belongs to the lid if that outside cell has y > y max
//  (this includes the lid's edges and corners), and to the no-slip walls otherwise.
//
//  The timed loop is stream/collide, halo exchange (between blocks only), then the wall sweeps:
//    pull: StreamCollide, streamPullSync, NoSlip, UBB
//    AA:   StreamCollideEven, aaPullSync, NoSlipEven, UBBEven, then
//          StreamCollideOdd,  aaPushSync, NoSlipOdd,  UBBOdd
//  The wall sweeps never run before the first step: InitPdfs leaves the slots of the PDFs that point
//  into a wall unwritten (they stay 0), and the first stream/collide reads the PDFs streamed in from the
//  walls from what InitPdfs placed there.
//
//  Environment:
//    SWEEPGEN_LDC_OUTPUT=last   only the last time step writes rho and u (default: every step)
//    SWEEPGEN_LDC_DUMP=<prefix> after the run, each process writes its cells to <prefix>.rank<r>.bin
//                               (verification only; format in writeDump)

#include "blockforest/all.h"

#include "core/all.h"

#include "stencil/all.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "gen/SweepgenLdcCPU.hpp"
#include "walberla/V8.hpp"

#ifndef SWEEPGEN_LDC_VARIANT
#   define SWEEPGEN_LDC_VARIANT "unknown"
#endif
#ifndef SWEEPGEN_LDC_COMPILER
#   define SWEEPGEN_LDC_COMPILER "unknown"
#endif

namespace SweepgenLdcCPU
{

using namespace walberla;
using namespace walberla::v8;

using MemoryTag = memtag::host;

using ScalarField_T = memory::Field< real_t, 1, MemoryTag >;
using VectorField_T = memory::Field< real_t, 3, MemoryTag >;

using LbStencil  = stencil::D3Q19;
using PdfField_T = memory::Field< real_t, LbStencil::Q, MemoryTag >;

//  Physics of waLBerla's UniformGridCPU LDC scenario: lid velocity (UniformGridCPU.py, UBB([0.05, 0, 0]))
//  and relaxation rate (simulation_setup/benchmark_configs.py, Scenario(omega=1.8))
constexpr double LID_VELOCITY{ 0.05 };
constexpr double OMEGA{ 1.8 };

/// splitmix64 finalizer: a 64-bit mixing function
inline uint64_t mix64(uint64_t x)
{
   x += 0x9E3779B97F4A7C15ULL;
   x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
   x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
   return x ^ (x >> 31);
}

/// Bit pattern of a value; +0 and -0 give the same pattern (they compare equal)
inline uint64_t valueBits(real_t v)
{
   using Bits_T = std::conditional_t< sizeof(real_t) == 8, uint64_t, uint32_t >;
   if (std::fpclassify(v) == FP_ZERO) return 0;
   return uint64_t(std::bit_cast< Bits_T >(v));
}

void run(int argc, char** argv)
{
   mpi::Environment env{ argc, argv };

   const uint_t cells     = argc > 1 ? uint_c(std::atoi(argv[1])) : uint_t(64);
   const uint_t timesteps = argc > 2 ? uint_c(std::atoi(argv[2])) : uint_t(200);
   const uint_t warmup    = argc > 3 ? uint_c(std::atoi(argv[3])) : uint_t(10);

   //  Closed box, one block of cells^3 per process, cell size 1
   const Vector3< uint_t > numBlocks{ math::getFactors3D(uint_c(MPIManager::instance()->numProcesses())) };
   auto blocks = blockforest::createUniformBlockGrid(numBlocks[0], numBlocks[1], numBlocks[2], cells, cells, cells,
                                                     real_t(1), /*oneBlockPerProcess*/ true,
                                                     /*periodic*/ false, false, false);
   const CellInterval domainBB{ blocks->getDomainCellBB() };

   //  Fluid at rest with density 1; the fields are initialized including their ghost layers
   ScalarField_T rho{ *blocks, uint_t(1), real_c(1.0) };
   VectorField_T u{ *blocks };
   PdfField_T pdfs{ *blocks };

   const real_t omega{ real_c(OMEGA) };
   const real_t uLid{ real_c(LID_VELOCITY) };

   const char* outputEnv{ std::getenv("SWEEPGEN_LDC_OUTPUT") };
   const bool outputEveryStep{ outputEnv == nullptr || std::strcmp(outputEnv, "last") != 0 };

   gen::LBM::InitPdfs lbInit{ pdfs, rho, u };
   for (auto& b : *blocks)
      lbInit(&b);

   //  ---- Walls: classify every link from a fluid cell into a cell outside the domain -------------
   uint64_t linksLid{ 0 };
   uint64_t linksNoSlip{ 0 };
   auto globalWallCell = [&](const sweepgen::PotentialBoundaryLink& link) {
      Cell w{ link.wallCell };
      blocks->transformBlockLocalToGlobalCell(w, link.block);
      return w;
   };
   auto isLid = [&](const sweepgen::PotentialBoundaryLink& link) -> bool {
      const Cell w{ globalWallCell(link) };
      const bool lid{ w.y() > domainBB.yMax() };
      linksLid += lid ? 1 : 0;
      return lid;
   };
   auto isNoSlip = [&](const sweepgen::PotentialBoundaryLink& link) -> bool {
      const Cell w{ globalWallCell(link) };
      const bool wall{ !domainBB.contains(w) && !(w.y() > domainBB.yMax()) };
      linksNoSlip += wall ? 1 : 0;
      return wall;
   };

   WcTimer tBulk;
   WcTimer tComm;
   WcTimer tBc;

#if defined(SWEEPGEN_LDC_AA)
   //  In-place AA streaming (see SweepgenLbmCPU.cpp): even and odd sweeps alternate on the one PDF field,
   //  starting with an even sweep on the PDFs InitPdfs wrote. After each sweep the wall sweeps of the
   //  same parity bounce the PDFs that left through a wall back into the domain.
   gen::LBM::StreamCollideEven streamCollideEven{ pdfs, rho, u, omega };
   gen::LBM::StreamCollideOdd streamCollideOdd{ pdfs, rho, u, omega };
   gen::LBM::StreamCollideEvenNoOutput streamCollideEvenNoOutput{ pdfs, omega };
   gen::LBM::StreamCollideOddNoOutput streamCollideOddNoOutput{ pdfs, omega };
#   if defined(SWEEPGEN_LDC_SPLIT)
   //  The direction-split stream/collide (SweepgenLdcCPU.py --split) instead of the sweeps above
   auto streamCollideSplit = [&](IBlock& b, bool odd, bool output) {
      const memory::BufferView< real_t, 4, MemoryTag > fv{ pdfs.bufferSystem().view(b) };
      const memory::BufferView< real_t, 4, MemoryTag > rv{ rho.bufferSystem().view(b) };
      const memory::BufferView< real_t, 4, MemoryTag > uv{ u.bufferSystem().view(b) };
      gen::LBM_SPLIT::streamCollide(fv.data(), rv.data(), uv.data(), int64_t(fv.shape()[0]), int64_t(fv.shape()[1]),
                                    int64_t(fv.shape()[2]), int64_t(fv.strides()[1]), int64_t(fv.strides()[2]),
                                    int64_t(fv.strides()[3]), int64_t(rv.strides()[1]), int64_t(rv.strides()[2]),
                                    int64_t(uv.strides()[1]), int64_t(uv.strides()[2]), int64_t(uv.strides()[3]), odd,
                                    output, omega);
   };
#   endif
   auto noSlipEven = gen::NoSlipEvenFactory{ blocks, pdfs }.fromLinks(isNoSlip);
   auto ubbEven    = gen::UBBEvenFactory{ blocks, pdfs, uLid }.fromLinks(isLid);
   auto noSlipOdd  = gen::NoSlipOddFactory{ blocks, pdfs }.fromLinks(isNoSlip);
   auto ubbOdd     = gen::UBBOddFactory{ blocks, pdfs, uLid }.fromLinks(isLid);
   //  Each link was counted once per parity
   linksLid /= 2;
   linksNoSlip /= 2;

   auto aaPull =
      HaloExchange::create< LbStencil, MemoryTag >(blocks).sync(halo_exchange::aaPullSync< LbStencil >(pdfs)).build();
   auto aaPush =
      HaloExchange::create< LbStencil, MemoryTag >(blocks).sync(halo_exchange::aaPushSync< LbStencil >(pdfs)).build();
   bool oddStep{ false };

   auto timestep = [&](bool output) {
      tBulk.start();
      for (auto& b : *blocks)
      {
#   if defined(SWEEPGEN_LDC_SPLIT)
         streamCollideSplit(b, oddStep, output);
#   else
         if (oddStep) { output ? streamCollideOdd(&b) : streamCollideOddNoOutput(&b); }
         else { output ? streamCollideEven(&b) : streamCollideEvenNoOutput(&b); }
#   endif
      }
      tBulk.end();
      tComm.start();
      if (oddStep) { aaPush(); }
      else { aaPull(); }
      tComm.end();
      tBc.start();
      for (auto& b : *blocks)
      {
         if (oddStep)
         {
            noSlipOdd(&b);
            ubbOdd(&b);
         }
         else
         {
            noSlipEven(&b);
            ubbEven(&b);
         }
      }
      tBc.end();
      oddStep = !oddStep;
      sweep::sync();
   };
#else
   gen::LBM::StreamCollide streamCollide{ pdfs, rho, u, omega };
   gen::LBM::StreamCollideNoOutput streamCollideNoOutput{ pdfs, omega };
   auto noSlip = gen::NoSlipFactory{ blocks, pdfs }.fromLinks(isNoSlip);
   auto ubb    = gen::UBBFactory{ blocks, pdfs, uLid }.fromLinks(isLid);
   auto haloExchange =
      HaloExchange::create< LbStencil, MemoryTag >(blocks).sync(halo_exchange::streamPullSync< LbStencil >(pdfs)).build();

   auto timestep = [&](bool output) {
      tBulk.start();
      for (auto& b : *blocks)
      {
         output ? streamCollide(&b) : streamCollideNoOutput(&b);
      }
      tBulk.end();
      tComm.start();
      haloExchange();
      tComm.end();
      tBc.start();
      for (auto& b : *blocks)
      {
         noSlip(&b);
         ubb(&b);
      }
      tBc.end();
      sweep::sync();
   };
#endif

   //  Every link into the lid comes from a cell of the top layer, in one of the 5 directions with c_y = +1
   mpi::allReduceInplace(linksLid, mpi::SUM);
   mpi::allReduceInplace(linksNoSlip, mpi::SUM);
   WALBERLA_CHECK_EQUAL(linksLid, uint64_t(5) * uint64_t(domainBB.xSize()) * uint64_t(domainBB.zSize()))

   for (uint_t t = 0; t < warmup; ++t)
      timestep(outputEveryStep);
   tBulk.reset();
   tComm.reset();
   tBc.reset();

   WcTimer timer;
   WALBERLA_MPI_BARRIER()
   timer.start();
   for (uint_t t = 0; t < timesteps; ++t)
      timestep(outputEveryStep || t + 1 == timesteps);
   WALBERLA_MPI_BARRIER()
   timer.end();

   //  ---- Checksums of the last step's density and velocity fields ---------------------------------
   //  energy and mass: sums in double precision. fieldhash: XOR over all cells of a hash of the global
   //  cell and the bit patterns of rho, u_x, u_y, u_z, so it depends neither on the order nor on the
   //  block decomposition. Equal hashes mean bit-identical fields (with +0 == -0).
   double energy{ 0 };
   double mass{ 0 };
   uint64_t fieldHash{ 0 };
   const CellInterval interior{ { 0, 0, 0 }, { cell_idx_c(cells) - 1, cell_idx_c(cells) - 1, cell_idx_c(cells) - 1 } };
   for (auto& b : *blocks)
   {
      memory::FieldView< real_t, 1, MemoryTag > rhoView{ rho, b };
      memory::FieldView< real_t, 3, MemoryTag > uView{ u, b };
      sweep::forAllCells(exectag::Serial{}, interior, [&](Cell c) {
         const double ux{ uView(c, 0) }, uy{ uView(c, 1) }, uz{ uView(c, 2) };
         energy += ux * ux + uy * uy + uz * uz;
         mass += double(rhoView(c));
         Cell g{ c };
         blocks->transformBlockLocalToGlobalCell(g, b);
         uint64_t h{ mix64((uint64_t(uint32_t(g.x())) << 42) ^ (uint64_t(uint32_t(g.y())) << 21) ^
                           uint64_t(uint32_t(g.z()))) };
         h = mix64(h ^ valueBits(rhoView(c)));
         h = mix64(h ^ valueBits(uView(c, 0)));
         h = mix64(h ^ valueBits(uView(c, 1)));
         h = mix64(h ^ valueBits(uView(c, 2)));
         fieldHash ^= h;
      });
   }
   mpi::allReduceInplace(energy, mpi::SUM);
   mpi::allReduceInplace(mass, mpi::SUM);
   mpi::allReduceInplace(fieldHash, mpi::BITWISE_XOR);

   //  ---- Optional dump for the verification scripts ------------------------------------------------
   //  <prefix>.rank<r>.bin: "LDCDUMP1", uint32 sizeof(real_t), uint32 0, uint64 number of cells, then per
   //  cell int32 x, y, z (global) and double rho, u_x, u_y, u_z
   if (const char* dumpPrefix = std::getenv("SWEEPGEN_LDC_DUMP"))
   {
      std::ostringstream name;
      name << dumpPrefix << ".rank" << MPIManager::instance()->rank() << ".bin";
      std::ofstream out(name.str(), std::ios::binary);
      out.write("LDCDUMP1", 8);
      const uint32_t realBytes{ uint32_t(sizeof(real_t)) }, pad{ 0 };
      uint64_t n{ 0 };
      for (auto& b : *blocks)
      {
         (void) b;
         n += uint64_t(interior.numCells());
      }
      out.write(reinterpret_cast< const char* >(&realBytes), 4);
      out.write(reinterpret_cast< const char* >(&pad), 4);
      out.write(reinterpret_cast< const char* >(&n), 8);
      for (auto& b : *blocks)
      {
         memory::FieldView< real_t, 1, MemoryTag > rhoView{ rho, b };
         memory::FieldView< real_t, 3, MemoryTag > uView{ u, b };
         sweep::forAllCells(exectag::Serial{}, interior, [&](Cell c) {
            Cell g{ c };
            blocks->transformBlockLocalToGlobalCell(g, b);
            const int32_t xyz[3]{ int32_t(g.x()), int32_t(g.y()), int32_t(g.z()) };
            const double vals[4]{ double(rhoView(c)), double(uView(c, 0)), double(uView(c, 1)), double(uView(c, 2)) };
            out.write(reinterpret_cast< const char* >(xyz), sizeof(xyz));
            out.write(reinterpret_cast< const char* >(vals), sizeof(vals));
         });
      }
      WALBERLA_CHECK(out.good(), "writing " << name.str() << " failed")
   }

   const uint_t totalCells{ domainBB.numCells() };
   const double seconds{ timer.last() };
   WALBERLA_ROOT_SECTION()
   {
      std::ostringstream hash;
      hash << "0x" << std::hex << std::setw(16) << std::setfill('0') << fieldHash;
      std::cout << std::setprecision(17);
      std::cout << "SweepgenLdcCPU variant=" << SWEEPGEN_LDC_VARIANT << " compiler=" << SWEEPGEN_LDC_COMPILER
                << " real_t=" << sizeof(real_t) * 8 << " blocks=" << numBlocks[0] << "x" << numBlocks[1] << "x"
                << numBlocks[2] << " domain=" << domainBB.xSize() << "x" << domainBB.ySize() << "x" << domainBB.zSize()
                << " cells=" << totalCells << " steps=" << timesteps << " warmup=" << warmup
                << " output=" << (outputEveryStep ? "every" : "last") << " omega=" << OMEGA
                << " u_lid=" << LID_VELOCITY << " seconds=" << seconds
                << " mlups=" << double(totalCells * timesteps) / seconds / 1e6 << " energy=" << energy
                << " mass=" << mass << " fieldhash=" << hash.str() << " links_noslip=" << linksNoSlip
                << " links_ubb=" << linksLid << " t_bulk=" << tBulk.total() << " t_comm=" << tComm.total()
                << " t_bc=" << tBc.total() << std::endl;
   }
}

} // namespace SweepgenLdcCPU

int main(int argc, char** argv)
{
   SweepgenLdcCPU::run(argc, argv);
   return EXIT_SUCCESS;
}
