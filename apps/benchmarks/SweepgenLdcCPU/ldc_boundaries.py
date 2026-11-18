# This file is part of waLBerla. waLBerla is free software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# waLBerla is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License along
# with waLBerla (see COPYING.txt). If not, see <http://www.gnu.org/licenses/>.

"""Sparse boundary sweeps for the in-place AA streaming pattern (SweepgenLdcCPU).

sweepgen's `SparseBoundary` always indexes the PDFs as for the `pull` pattern
(`sweepgen/boundaries/sparse.py`, `_get_kernel_assignments`: `streaming_pattern = "pull"`,
`prev_timestep = Timestep.BOTH`). With the AA pattern the PDFs that a boundary reads and writes
sit in different slots after an even and after an odd time step. lbmpy describes both cases with
`BetweenTimestepsIndexing(..., prev_timestep=EVEN|ODD, streaming_pattern="aa")`.

`AASparseBoundary` is `SparseBoundary` with exactly that one difference: `_get_kernel_assignments`
below is a copy of the method in sparse.py with the streaming pattern and the previous time step
as parameters. Everything else (index structs, factory, sweep class) is inherited unchanged. Use
one instance per time step parity, with distinct names (e.g. NoSlipEven, NoSlipOdd), and run the
`Even` sweep after every even stream/collide sweep and the `Odd` one after every odd sweep.
"""

from __future__ import annotations

from pystencils import Assignment, Field

from lbmpy.advanced_streaming import Timestep
from lbmpy.advanced_streaming.indexing import BetweenTimestepsIndexing
from lbmpy.boundaries.boundaryconditions import LbBoundary
from lbmpy.methods import AbstractLbMethod

from sweepgen.boundaries.sparse import (
    SparseBoundary,
    SparseBoundaryDefinition,
    LbmpyBoundaryWrapper,
    BoundaryIndexType,
)


class AASparseBoundary(SparseBoundary):
    """`SparseBoundary` for the AA pattern, applied after a time step of the given parity."""

    def __init__(
        self,
        bc: LbBoundary | SparseBoundaryDefinition,
        lb_method: AbstractLbMethod,
        pdf_field: Field,
        prev_timestep: Timestep,
    ):
        if prev_timestep not in (Timestep.EVEN, Timestep.ODD):
            raise ValueError("prev_timestep must be Timestep.EVEN or Timestep.ODD")
        super().__init__(bc, lb_method, pdf_field)
        self._prev_timestep = prev_timestep

    def _get_kernel_assignments(self) -> list[Assignment]:
        #   Copy of SparseBoundary._get_kernel_assignments; only the two indexing
        #   parameters differ (pull/BOTH there, aa/<parity> here)
        index_field = self._index_field()

        indexing = BetweenTimestepsIndexing(
            self._pdf_field,
            self._lb_method.stencil,
            self._prev_timestep,
            "aa",
            BoundaryIndexType,
            BoundaryIndexType,
        )

        f_out, f_in = indexing.proxy_fields
        dir_symbol = indexing.dir_symbol
        inv_dir = indexing.inverse_dir_symbol

        boundary_assignments = self._bc(
            f_out,
            f_in,
            dir_symbol,
            inv_dir,
            self._lb_method,
            index_field,
            None,
        )
        boundary_assignments = indexing.substitute_proxies(boundary_assignments)

        elements: list[Assignment] = []

        index_arrs_node = indexing.create_code_node()
        elements += index_arrs_node.get_array_declarations()

        if isinstance(self._bc, LbmpyBoundaryWrapper):
            for node in self._bc.lbmpy_object.get_additional_code_nodes(self._lb_method)[::-1]:
                elements += node.get_array_declarations()

        elements += [Assignment(dir_symbol, index_field[0]("dir"))]
        elements += boundary_assignments.all_assignments

        return elements
