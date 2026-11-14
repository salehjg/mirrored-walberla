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
//! \file GenericFieldPackInfo.hpp
//! \author Frederik Hennig <frederik.hennig@fau.de>
//
//======================================================================================================================

#pragma once

#include "core/Stdlib.hpp"
#include WALBERLA_STDLIB(span)

#include "stencil/Directions.h"

#include "../HaloExchange.hpp"
#include "walberla/v8/StencilRanges.hpp"
#include "walberla/v8/Sweep.hpp"
#include "walberla/v8/memory/Field.hpp"
#include "walberla/v8/memory/MemoryTags.hpp"

namespace walberla::v8::halo_exchange
{

namespace fields
{

/**
 * Helpers of the stencil-aware pack infos (GenericStreamPullPackInfo, GenericAAPullPackInfo,
 * GenericAAPushPackInfo). These move one PDF slot at a time: `pack`, `unpack` and `localCopy` loop over the
 * slots outside and over the cells inside, and a buffer holds one contiguous segment per slot. With fzyx
 * fields every pass then reads and writes a single slot slice in storage order. A loop over the slots per
 * cell touches up to five slot slices per cell instead, and on an x face (one cell per row) a separate cache
 * line for every value: on the SpacemiT X60 the exchanges took 21 % of an AA time step on one core that way.
 */
namespace detail
{

/// Position of `cell` in the z-y-x order of `region`
WALBERLA_HOST_DEVICE inline size_t cellIndex(const CellInterval& region, const Cell& cell)
{
   const Cell rc{ cell - region.min() };
   return size_t(rc.x() + cell_idx_c(region.xSize()) * (rc.y() + cell_idx_c(region.ySize()) * rc.z()));
}

/// Writes slot `q` of the cells of `region` to `segment`, in z-y-x order
template< sweep::ExecTag XTag, typename TView >
void packSlot(XTag xtag, const TView& view, const CellInterval& region, cell_idx_t q,
              typename TView::value_type* segment)
{
   sweep::forAllCells(xtag, region, [=] WALBERLA_HOST_DEVICE (Cell cell) {
      segment[cellIndex(region, cell)] = view(cell, q);
   });
}

/// Reads slot `q` of the cells `cells`, a part of `region`, from `segment`, which holds `region` in z-y-x order
template< sweep::ExecTag XTag, typename TView >
void unpackSlot(XTag xtag, const TView& view, const CellInterval& region, const CellInterval& cells, cell_idx_t q,
                const typename TView::value_type* segment)
{
   if (cells.empty()) { return; }
   sweep::forAllCells(xtag, cells, [=] WALBERLA_HOST_DEVICE (Cell cell) {
      view(cell, q) = segment[cellIndex(region, cell)];
   });
}

/// Copies slot `q` of the cells `src` to the cells `dst` (same size, paired in z-y-x order)
template< sweep::ExecTag XTag, typename TSrcView, typename TDstView >
void copySlot(XTag xtag, const TSrcView& srcView, const CellInterval& src, const TDstView& dstView,
              const CellInterval& dst, cell_idx_t q)
{
   if (src.empty()) { return; }
   sweep::forAllCellPairs(xtag, src, dst, [=] WALBERLA_HOST_DEVICE (Cell srcCell, Cell dstCell) {
      dstView(dstCell, q) = srcView(srcCell, q);
   });
}

} // namespace detail

template< memory::IField TField, sweep::ExecTag XTag >
   requires(!TField::IS_CONST)
class GenericFieldPackInfo
{
 public:
   using FieldType                = TField;
   using value_type               = typename FieldType::value_type;
   static constexpr size_t F_SIZE = FieldType::F_SIZE;

   /**
    * @param field The field whose ghost layers to exchange
    * @param numGhostLayers Number of ghost layers to exchange. Must not exceed the field's total number of ghost
    * layers.
    */
   GenericFieldPackInfo(const FieldType& field, size_t numGhostLayers)
      : field_{ field }, numGhostLayers_{ numGhostLayers }
   {
      if (numGhostLayers_ == 0 || numGhostLayers_ > field.numGhostLayers())
      {
         throw std::invalid_argument{ "Invalid number of ghost layers" };
      }
   }

   explicit GenericFieldPackInfo(const FieldType& field) : GenericFieldPackInfo(field, field.numGhostLayers()) {}

   void pack(const IBlock& srcBlock, stencil::Direction commDir, stdlib::span< value_type > buffer, XTag xtag) const
   {
      //  Explicit type instead of deducing the alias template `ConstFieldView`:
      //  Clang < 19 does not implement C++20 alias template argument deduction
      const memory::ConstFieldViewType< FieldType > srcView{ field_, srcBlock };
      const CellInterval srcRegion{ srcView.slices().borderSlice(commDir, numGhostLayers_) };

      constexpr size_t valuesPerCell{ F_SIZE };

      // TODO place mdspan over buffer for md-indexing
      // TODO Honor memory layout of field (fzyx or zyxf)

      sweep::forAllCells(xtag, srcRegion, [=] WALBERLA_HOST_DEVICE (Cell cell) {
         const Cell rc{ cell - srcRegion.min() };
         const size_t segmentStart(
            valuesPerCell * (uint_c(rc.x()) + srcRegion.xSize() * (uint_c(rc.y()) + srcRegion.ySize() * uint_c(rc.z())))
         );

         for (cell_idx_t q = 0; q < cell_idx_t(F_SIZE); ++q)
         {
            buffer[segmentStart + size_t(q)] = srcView(cell, q);
         }
      });
   }

   void unpack(IBlock& dstBlock, stencil::Direction commDir, stdlib::span< const value_type > buffer, XTag xtag) const
   {
      const FieldView dstView{ field_, dstBlock };
      const CellInterval dstRegion{ dstView.slices().ghostSlice(stencil::inverseDir[commDir], numGhostLayers_) };

      constexpr size_t valuesPerCell{ F_SIZE };

      sweep::forAllCells(xtag, dstRegion, [=] WALBERLA_HOST_DEVICE (Cell cell) {
         const Cell rc{ cell - dstRegion.min() };
         const size_t segmentStart(
            valuesPerCell * (uint_c(rc.x()) + dstRegion.xSize() * (uint_c(rc.y()) + dstRegion.ySize() * uint_c(rc.z())))
         );

         for (cell_idx_t q = 0; q < cell_idx_t(F_SIZE); ++q)
         {
            dstView(cell, q) = buffer[segmentStart + size_t(q)];
         }
      });
   }

   void localCopy(const IBlock& srcBlock, stencil::Direction commDir, IBlock& dstBlock, XTag xtag) const
   {
      //  Explicit type instead of deducing the alias template `ConstFieldView`:
      //  Clang < 19 does not implement C++20 alias template argument deduction
      const memory::ConstFieldViewType< FieldType > srcView{ field_, srcBlock };
      const CellInterval srcRegion{ srcView.slices().borderSlice(commDir, numGhostLayers_) };

      const FieldView dstView{ field_, dstBlock };
      const CellInterval dstRegion{ dstView.slices().ghostSlice(stencil::inverseDir[commDir], numGhostLayers_) };

      sweep::forAllCellPairs(xtag, srcRegion, dstRegion, [=] WALBERLA_HOST_DEVICE (Cell srcCell, Cell dstCell) {
         for (cell_idx_t q = 0; q < cell_idx_t(F_SIZE); ++q)
         {
            dstView(dstCell, q) = srcView(srcCell, q);
         }
      });
   }

   size_t packetSize(stencil::Direction commDir) const
   {
      size_t numCells{ field_.slices().borderSlice(commDir, numGhostLayers_).numCells() };
      return numCells * F_SIZE;
   }

 private:
   FieldType field_;
   size_t numGhostLayers_;
};

template< IField TField, typename TStencil, sweep::ExecTag XTag >
   requires(TField::F_SIZE == TStencil::Q)
class GenericStreamPullPackInfo
{
 public:
   using FieldType  = TField;
   using Stencil    = TStencil;
   using value_type = typename FieldType::value_type;

   /**
    * @param field The field whose ghost layers to exchange
    * @param numGhostLayers Number of ghost layers to exchange. Must not exceed the field's total number of ghost
    * layers.
    */
   explicit GenericStreamPullPackInfo(const FieldType& field) : field_{ field }
   {
      if (field.numGhostLayers() == 0) { throw std::invalid_argument{ "Field must have at least one ghost layer" }; }
   }

   void pack(const IBlock& srcBlock, stencil::Direction commDir, stdlib::span< value_type > buffer, XTag xtag) const
   {
      //  Explicit type instead of deducing the alias template `ConstFieldView`:
      //  Clang < 19 does not implement C++20 alias template argument deduction
      const memory::ConstFieldViewType< FieldType > srcView{ field_, srcBlock };
      const CellInterval srcRegion{ srcView.slices().borderSlice(commDir, 1) };

      value_type* segment{ buffer.data() };
      for (cell_idx_t q : stencil_ranges::subdirections< Stencil >(commDir))
      {
         detail::packSlot(xtag, srcView, srcRegion, q, segment);
         segment += srcRegion.numCells();
      }
   }

   void unpack(IBlock& dstBlock, stencil::Direction commDir, stdlib::span< const value_type > buffer, XTag xtag) const
   {
      const FieldView dstView{ field_, dstBlock };
      const CellInterval dstRegion{ dstView.slices().ghostSlice(stencil::inverseDir[commDir], 1) };

      const value_type* segment{ buffer.data() };
      for (cell_idx_t q : stencil_ranges::subdirections< Stencil >(commDir))
      {
         detail::unpackSlot(xtag, dstView, dstRegion, dstRegion, q, segment);
         segment += dstRegion.numCells();
      }
   }

   void localCopy(const IBlock& srcBlock, stencil::Direction commDir, IBlock& dstBlock, XTag xtag) const
   {
      //  Explicit type instead of deducing the alias template `ConstFieldView`:
      //  Clang < 19 does not implement C++20 alias template argument deduction
      const memory::ConstFieldViewType< FieldType > srcView{ field_, srcBlock };
      const CellInterval srcRegion{ srcView.slices().borderSlice(commDir, 1) };

      const FieldView dstView{ field_, dstBlock };
      const CellInterval dstRegion{ dstView.slices().ghostSlice(stencil::inverseDir[commDir], 1) };

      for (cell_idx_t q : stencil_ranges::subdirections< Stencil >(commDir))
      {
         detail::copySlot(xtag, srcView, srcRegion, dstView, dstRegion, q);
      }
   }

   size_t packetSize(stencil::Direction commDir) const
   {
      size_t numCells{ field_.slices().borderSlice(commDir, 1).numCells() };
      return numCells * stencil_ranges::subdirections< Stencil >(commDir).size();
   }

 private:
   FieldType field_;
};

/**
 * Ghost-layer exchange before the odd time step of the in-place AA streaming pattern.
 *
 * In the odd AA step, cell x reads slot inv(i) at x - c_i. For a cell next to the block border, x - c_i
 * may be a ghost cell, which must then hold slot inv(i) of the neighbor block's border cell, for every
 * direction i that points from that neighbor into this block. These are the inverse slots of those a
 * pull-streaming exchange (GenericStreamPullPackInfo) sends.
 */
template< IField TField, typename TStencil, sweep::ExecTag XTag >
   requires(TField::F_SIZE == TStencil::Q)
class GenericAAPullPackInfo
{
 public:
   using FieldType  = TField;
   using Stencil    = TStencil;
   using value_type = typename FieldType::value_type;

   explicit GenericAAPullPackInfo(const FieldType& field) : field_{ field }
   {
      if (field.numGhostLayers() == 0) { throw std::invalid_argument{ "Field must have at least one ghost layer" }; }
   }

   void pack(const IBlock& srcBlock, stencil::Direction commDir, stdlib::span< value_type > buffer, XTag xtag) const
   {
      const memory::ConstFieldViewType< FieldType > srcView{ field_, srcBlock };
      const CellInterval srcRegion{ srcView.slices().borderSlice(commDir, 1) };

      value_type* segment{ buffer.data() };
      for (stencil::Direction d : stencil_ranges::subdirections< Stencil >(commDir))
      {
         detail::packSlot(xtag, srcView, srcRegion, cell_idx_c(Stencil::invDirIdx(d)), segment);
         segment += srcRegion.numCells();
      }
   }

   void unpack(IBlock& dstBlock, stencil::Direction commDir, stdlib::span< const value_type > buffer, XTag xtag) const
   {
      const FieldView dstView{ field_, dstBlock };
      const CellInterval dstRegion{ dstView.slices().ghostSlice(stencil::inverseDir[commDir], 1) };

      const value_type* segment{ buffer.data() };
      for (stencil::Direction d : stencil_ranges::subdirections< Stencil >(commDir))
      {
         detail::unpackSlot(xtag, dstView, dstRegion, dstRegion, cell_idx_c(Stencil::invDirIdx(d)), segment);
         segment += dstRegion.numCells();
      }
   }

   void localCopy(const IBlock& srcBlock, stencil::Direction commDir, IBlock& dstBlock, XTag xtag) const
   {
      const memory::ConstFieldViewType< FieldType > srcView{ field_, srcBlock };
      const CellInterval srcRegion{ srcView.slices().borderSlice(commDir, 1) };

      const FieldView dstView{ field_, dstBlock };
      const CellInterval dstRegion{ dstView.slices().ghostSlice(stencil::inverseDir[commDir], 1) };

      for (stencil::Direction d : stencil_ranges::subdirections< Stencil >(commDir))
      {
         detail::copySlot(xtag, srcView, srcRegion, dstView, dstRegion, cell_idx_c(Stencil::invDirIdx(d)));
      }
   }

   size_t packetSize(stencil::Direction commDir) const
   {
      size_t numCells{ field_.slices().borderSlice(commDir, 1).numCells() };
      return numCells * stencil_ranges::subdirections< Stencil >(commDir).size();
   }

 private:
   FieldType field_;
};

/**
 * Ghost-layer exchange after the odd time step of the in-place AA streaming pattern.
 *
 * In the odd AA step, cell x writes its post-collision value of direction i to slot i at x + c_i. For a
 * cell next to the block border, x + c_i may be a ghost cell, whose slot i then belongs to the neighbor
 * block that owns that cell. This exchange sends those slots (directions i that point from this block
 * into the neighbor) from the ghost layer back to the neighbor's border cells.
 *
 * A ghost slice also holds slots that no interior cell of the sending block wrote in that step: at its
 * edges, the writer g - c_i lies outside the sending block. The owner receives those slots from another
 * neighbor (over an edge instead of a face), so they are skipped when unpacking.
 */
template< IField TField, typename TStencil, sweep::ExecTag XTag >
   requires(TField::F_SIZE == TStencil::Q)
class GenericAAPushPackInfo
{
 public:
   using FieldType  = TField;
   using Stencil    = TStencil;
   using value_type = typename FieldType::value_type;

   explicit GenericAAPushPackInfo(const FieldType& field) : field_{ field }
   {
      if (field.numGhostLayers() == 0) { throw std::invalid_argument{ "Field must have at least one ghost layer" }; }
   }

   void pack(const IBlock& srcBlock, stencil::Direction commDir, stdlib::span< value_type > buffer, XTag xtag) const
   {
      const memory::ConstFieldViewType< FieldType > srcView{ field_, srcBlock };
      const CellInterval srcRegion{ srcView.slices().ghostSlice(commDir, 1) };

      value_type* segment{ buffer.data() };
      for (stencil::Direction d : stencil_ranges::subdirections< Stencil >(commDir))
      {
         detail::packSlot(xtag, srcView, srcRegion, cell_idx_c(Stencil::idx[d]), segment);
         segment += srcRegion.numCells();
      }
   }

   void unpack(IBlock& dstBlock, stencil::Direction commDir, stdlib::span< const value_type > buffer, XTag xtag) const
   {
      const FieldView dstView{ field_, dstBlock };
      const CellInterval dstRegion{ dstView.slices().borderSlice(stencil::inverseDir[commDir], 1) };
      //  All blocks share the field's shape, so the sender's ghost slice has these coordinates in its block
      const CellInterval srcRegion{ field_.slices().ghostSlice(commDir, 1) };
      const CellInterval interior{ field_.slices().interior() };
      const Cell toDst{ dstRegion.min() - srcRegion.min() };

      const value_type* segment{ buffer.data() };
      for (stencil::Direction d : stencil_ranges::subdirections< Stencil >(commDir))
      {
         CellInterval dstCells{ writtenGhosts(interior, srcRegion, d) };
         dstCells.shift(toDst);
         detail::unpackSlot(xtag, dstView, dstRegion, dstCells, cell_idx_c(Stencil::idx[d]), segment);
         segment += dstRegion.numCells();
      }
   }

   void localCopy(const IBlock& srcBlock, stencil::Direction commDir, IBlock& dstBlock, XTag xtag) const
   {
      const memory::ConstFieldViewType< FieldType > srcView{ field_, srcBlock };
      const CellInterval srcRegion{ srcView.slices().ghostSlice(commDir, 1) };

      const FieldView dstView{ field_, dstBlock };
      const CellInterval dstRegion{ dstView.slices().borderSlice(stencil::inverseDir[commDir], 1) };
      const CellInterval interior{ field_.slices().interior() };
      const Cell toDst{ dstRegion.min() - srcRegion.min() };

      for (stencil::Direction d : stencil_ranges::subdirections< Stencil >(commDir))
      {
         const CellInterval srcCells{ writtenGhosts(interior, srcRegion, d) };
         CellInterval dstCells{ srcCells };
         dstCells.shift(toDst);
         detail::copySlot(xtag, srcView, srcCells, dstView, dstCells, cell_idx_c(Stencil::idx[d]));
      }
   }

   size_t packetSize(stencil::Direction commDir) const
   {
      size_t numCells{ field_.slices().ghostSlice(commDir, 1).numCells() };
      return numCells * stencil_ranges::subdirections< Stencil >(commDir).size();
   }

 private:
   //  The cells of the ghost slice `ghosts` whose slot d was written in the odd step, i.e. by an interior cell
   //  g - c_d: the interior shifted by c_d, cut to the slice
   static CellInterval writtenGhosts(const CellInterval& interior, const CellInterval& ghosts, stencil::Direction d)
   {
      CellInterval written{ interior };
      written.shift(cell_idx_c(stencil::cx[d]), cell_idx_c(stencil::cy[d]), cell_idx_c(stencil::cz[d]));
      written.intersect(ghosts);
      return written;
   }

   FieldType field_;
};

} // namespace fields

} // namespace walberla::v8::halo_exchange
