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
//======================================================================================================================

#pragma once

#include "blockforest/all.h"

#include "core/all.h"

#include "stencil/all.h"

#include <cmath>

#include "gen/LbmAlgorithms.hpp"
#include "walberla/V8.hpp"
#include "walberla/v8/Testing.hpp"

namespace lbm_scenarios
{

using namespace walberla;
using namespace walberla::v8;

/**
 * The in-place AA streaming pattern (`LBM_AA`: StreamCollideEven/StreamCollideOdd, synchronized with
 * aaPullSync after even and aaPushSync after odd steps) must reproduce the pull pattern (`LBM`:
 * StreamCollide, streamPullSync) cell by cell, on a non-uniform, fully periodic flow, after even and odd
 * numbers of steps. Run with several process counts, it covers the exchanges between blocks as well as
 * the periodic exchange of a block with itself.
 */
template< memory::MemTag MemoryTag >
struct AAMatchesPull
{
   using ScalarField_T = memory::Field< real_t, 1, MemoryTag >;
   using VectorField_T = memory::Field< real_t, 3, MemoryTag >;
   using LbStencil     = stencil::D3Q19;
   using PdfField_T    = memory::Field< real_t, LbStencil::Q, MemoryTag >;

   static void run()
   {
      uint_t numProcesses = uint_c(MPIManager::instance()->numProcesses());
      Vector3< uint_t > numBlocks{ math::getFactors3D(numProcesses) };
      Vector3< uint_t > cellsPerBlock{ 6, 6, 6 };

      auto blocks = blockforest::createUniformBlockGrid(numBlocks[0], numBlocks[1], numBlocks[2], //
                                                        cellsPerBlock[0], cellsPerBlock[1], cellsPerBlock[2], //
                                                        1.0, true, true, true, true);

      ScalarField_T rho{ *blocks, 0UL, real_c(1.0) };
      VectorField_T u{ *blocks, 0UL, real_c(0.0) };
      PdfField_T pdfs{ *blocks };

      ScalarField_T rhoAA{ *blocks, 0UL, real_c(1.0) };
      VectorField_T uAA{ *blocks, 0UL, real_c(0.0) };
      PdfField_T pdfsAA{ *blocks };

      const Vector3< real_t > force{ 1e-4, -2e-4, 0. };
      const real_t omega{ 1.4 };

      //  A velocity field that varies along every axis, so that every exchanged slot matters
      const AABB domain{ blocks->getDomain() };
      const real_t pi2{ real_c(2.0 * math::pi) };
      for (auto& b : *blocks)
      {
         FieldView uView{ u, b };
         FieldView uAAView{ uAA, b };
         domain::BlockGeometry bGeom{ *blocks, b };

         sweep::forAllCells(exectag::Serial{}, *blocks, [&](Cell c) {
            const auto x{ bGeom.localGrid().cellCenter(c) };
            const real_t sx{ std::sin(pi2 * x[0] / domain.xSize()) };
            const real_t sy{ std::sin(pi2 * x[1] / domain.ySize()) };
            const real_t sz{ std::cos(pi2 * x[2] / domain.zSize()) };
            const real_t uc[3]{ real_c(0.03) * sy * sz, real_c(0.02) * sz * sx, real_c(0.01) * sx * sy };
            for (cell_idx_t i = 0; i < 3; ++i)
            {
               uView(c, i)   = uc[i];
               uAAView(c, i) = uc[i];
            }
         });
      }

      gen::LBM::InitPdfs init{ pdfs, rho, u, force };
      gen::LBM_AA::InitPdfs initAA{ pdfsAA, rhoAA, uAA, force };
      for (auto& b : *blocks)
      {
         init(&b);
         initAA(&b);
      }

      gen::LBM::StreamCollide streamCollide(pdfs, rho, u, force, omega);
      gen::LBM_AA::StreamCollideEven streamCollideEven(pdfsAA, rhoAA, uAA, force, omega);
      gen::LBM_AA::StreamCollideOdd streamCollideOdd(pdfsAA, rhoAA, uAA, force, omega);

      auto pullSync = HaloExchange::create< LbStencil, MemoryTag >(blocks)
                         .sync(halo_exchange::streamPullSync< LbStencil >(pdfs))
                         .build();
      auto aaPullSync = HaloExchange::create< LbStencil, MemoryTag >(blocks)
                           .sync(halo_exchange::aaPullSync< LbStencil >(pdfsAA))
                           .build();
      auto aaPushSync = HaloExchange::create< LbStencil, MemoryTag >(blocks)
                           .sync(halo_exchange::aaPushSync< LbStencil >(pdfsAA))
                           .build();

      for (uint_t t = 0; t < 9; ++t)
      {
         const bool odd{ t % 2 == 1 };
         for (auto& b : *blocks)
         {
            streamCollide(&b);
            if (odd) { streamCollideOdd(&b); }
            else { streamCollideEven(&b); }
         }
         pullSync();
         if (odd) { aaPushSync(); }
         else { aaPullSync(); }

         sweep::sync();

         for (auto& b : *blocks)
         {
            FieldView uView{ u, b };
            FieldView uAAView{ uAA, b };
            FieldView rhoView{ rho, b };
            FieldView rhoAAView{ rhoAA, b };

            sweep::forAllCells(exectag::Serial{}, *blocks, [&](Cell c) {
               testing::with_tolerance(1e-14, 1e-12).assert_close(rhoAAView(c), rhoView(c));
               for (cell_idx_t i = 0; i < 3; ++i)
               {
                  testing::with_tolerance(1e-14, 1e-12).assert_close(uAAView(c, i), uView(c, i));
               }
            });
         }
      }
   }
};
} // namespace lbm_scenarios
