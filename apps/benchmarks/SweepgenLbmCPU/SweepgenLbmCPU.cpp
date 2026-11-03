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
//! \file SweepgenLbmCPU.cpp
//! \brief Throughput (MLUP/s) of the DoubleShearLayer LBM sweeps, per code generation variant
//
//======================================================================================================================

//  The same app is built once per code generation variant (see CMakeLists.txt and
//  SweepgenLbmCPU.py). Each run prints one result line, so the variants can be compared
//  directly:
//
//    SweepgenLbmCPU variant=<name> real_t=<bits> cells=<n> steps=<n> seconds=<t> mlups=<x> energy=<e>
//
//  The kinetic energy at the end is a checksum: all variants must agree on it to rounding.
//
//  Usage: SweepgenLbmCPU_<variant> [cells per block and direction = 64] [timesteps = 200] [warmup = 10]
//  One block per MPI process. The timed loop is stream-collide plus halo exchange, as in a
//  production time step.

#include "blockforest/all.h"

#include "core/all.h"

#include "stencil/all.h"

#include <cstdlib>
#include <iostream>

#include "gen/SweepgenLbmCPU.hpp"
#include "walberla/V8.hpp"

#ifndef SWEEPGEN_LBM_VARIANT
#   define SWEEPGEN_LBM_VARIANT "unknown"
#endif

namespace SweepgenLbmCPU
{

using namespace walberla;
using namespace walberla::v8;

using MemoryTag = memtag::host;

using ScalarField_T = memory::Field< real_t, 1, MemoryTag >;
using VectorField_T = memory::Field< real_t, 3, MemoryTag >;

using LbStencil  = stencil::D3Q19;
using PdfField_T = memory::Field< real_t, LbStencil::Q, MemoryTag >;

void run(int argc, char** argv)
{
   mpi::Environment env{ argc, argv };

   const uint_t cells     = argc > 1 ? uint_c(std::atoi(argv[1])) : uint_t(64);
   const uint_t timesteps = argc > 2 ? uint_c(std::atoi(argv[2])) : uint_t(200);
   const uint_t warmup    = argc > 3 ? uint_c(std::atoi(argv[3])) : uint_t(10);

   //  Fully periodic domain, one block of cells^3 per process. y spans [0, 1] (the shear
   //  layers' coordinate); x and z use the same cell size.
   const Vector3< uint_t > numBlocks{ math::getFactors3D(uint_c(MPIManager::instance()->numProcesses())) };
   const real_t ny{ real_c(numBlocks[1] * cells) };
   const AABB domainAabb{ 0., 0., 0., real_c(numBlocks[0] * cells) / ny, 1., real_c(numBlocks[2] * cells) / ny };

   auto blocks = blockforest::createUniformBlockGrid(domainAabb, numBlocks[0], numBlocks[1], numBlocks[2], cells,
                                                     cells, cells, /*oneBlockPerProcess*/ true, true, true, true);

   ScalarField_T rho{ *blocks };
   VectorField_T u{ *blocks };
   PdfField_T pdfs{ *blocks };

   //  Physics of the DoubleShearLayer example (its .prm values)
   const real_t reynolds{ 30000 };
   const real_t kappa{ 80 };
   const real_t u_0{ 0.04 };
   const real_t delta{ 0.05 };
   const real_t nu{ (u_0 * real_c(blocks->getDomainCellBB().xSize())) / reynolds };
   //  real_c: braced initialization must not narrow double to float (single precision)
   const real_t tau{ real_c(nu / (1. / 3.)) };
   const real_t omega{ real_c(2. / (2. * tau + 1.)) };

   gen::SetInitialState setInitialState{ blocks, rho, u, delta, kappa, u_0 };
   gen::LBM::InitPdfs lbInit{ pdfs, rho, u };
   for (auto& b : *blocks)
   {
      setInitialState(&b);
      lbInit(&b);
   }

   gen::LBM::StreamCollide streamCollide{ pdfs, rho, u, omega };
   auto haloExchange =
      HaloExchange::create< LbStencil, MemoryTag >(blocks).sync(halo_exchange::streamPullSync< LbStencil >(pdfs)).build();

   //  The pull stream reads the ghost layers, so they must be filled before the first step
   haloExchange();

   auto timestep = [&]() {
      for (auto& b : *blocks)
      {
         streamCollide(&b);
      }
      haloExchange();
      sweep::sync();
   };

   for (uint_t t = 0; t < warmup; ++t)
      timestep();

   WcTimer timer;
   WALBERLA_MPI_BARRIER()
   timer.start();
   for (uint_t t = 0; t < timesteps; ++t)
      timestep();
   WALBERLA_MPI_BARRIER()
   timer.end();

   //  Checksum: total kinetic energy of the last step's velocity field, summed in double
   //  precision so that it stays meaningful in single-precision builds
   double energy{ 0 };
   const CellInterval interior{ { 0, 0, 0 }, { cell_idx_c(cells) - 1, cell_idx_c(cells) - 1, cell_idx_c(cells) - 1 } };
   for (auto& b : *blocks)
   {
      memory::FieldView< real_t, 3, MemoryTag > uView{ u, b };
      sweep::forAllCells(exectag::Serial{}, interior, [&](Cell c) {
         const double ux{ uView(c, 0) }, uy{ uView(c, 1) }, uz{ uView(c, 2) };
         energy += ux * ux + uy * uy + uz * uz;
      });
   }
   mpi::allReduceInplace(energy, mpi::SUM);

   const uint_t totalCells{ blocks->getDomainCellBB().numCells() };
   const double seconds{ timer.last() };
   WALBERLA_ROOT_SECTION()
   {
      std::cout.precision(12);
      std::cout << "SweepgenLbmCPU variant=" << SWEEPGEN_LBM_VARIANT << " real_t=" << sizeof(real_t) * 8
                << " cells=" << totalCells
                << " steps=" << timesteps << " seconds=" << seconds
                << " mlups=" << real_c(totalCells * timesteps) / seconds / 1e6 << " energy=" << energy << std::endl;
   }
}

} // namespace SweepgenLbmCPU

int main(int argc, char** argv)
{
   SweepgenLbmCPU::run(argc, argv);
   return EXIT_SUCCESS;
}
