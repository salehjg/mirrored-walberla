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

"""Direction-split stream/collide for the in-place AA pattern (SRT).

The monolithic AA sweep keeps 19 PDF streams and 4 output streams live in one loop. Cores with a
small hardware prefetcher (e.g. the SpacemiT X60 follows about 8 streams) then wait for DRAM on
most accesses. The split variant works on blocks of BY rows of one z-plane in three passes:

  1. moments, in groups of GM directions: delta_rho and the momentum, accumulated in a scratch
  2. macroscopic values: rho and u (into the scratch, and optionally into the output fields)
  3. collision, in groups of GP opposite-direction pairs: reload the PDFs (now in L1/L2), relax
     towards lbmpy's equilibrium, store. In place, a pair's write targets are its read sources,
     so both members are loaded before either is stored.

Every pass is an ordinary pystencils kernel built from lbmpy's method, conserved quantities and AA
accessors, so it is vectorized like any other sweep. A generated C++ driver calls the passes block
by block: the field pointers are moved to the block and the iteration space is (n0, rows, 1). The
scratch arrays are passed as fields of the block's shape.

Used by SweepgenLbmCPU.py (--split) and by the kernel lab of docs/notes3 (misc/x60-lab/lab/gensplit.py).
"""

from __future__ import annotations

import copy
import re

import sympy as sp
import pystencils as ps

from lbmpy import LBMConfig, create_lb_method
from lbmpy.advanced_streaming.utility import get_accessor, Timestep


class SplitAA:
    """Passes and driver of the split AA stream/collide.

    Args:
        lbm_config: method definition; must be SRT (one relaxation rate) with zero-centered PDFs
        dtype: element type, e.g. "float32"
        by, gm, gp: rows per block, directions per moment pass, opposite pairs per collision pass
        gen_config: pystencils configuration of the passes (target, vectorization, prefetch)
    """

    def __init__(self, lbm_config: LBMConfig, dtype: str, by: int, gm: int, gp: int, gen_config: ps.CreateKernelConfig):
        self.stencil = lbm_config.stencil
        self.method = create_lb_method(lbm_config)
        #   The collision f_i + omega (feq_i - f_i) of every direction on its own is SRT's
        if len(set(self.method.relaxation_rates)) != 1:
            raise ValueError("SplitAA needs an SRT method")
        self.omega = next(iter(set(self.method.relaxation_rates)))
        self.dtype, self.by, self.gm, self.gp = dtype, by, gm, gp
        self.gen_config = gen_config

    def stage_rules(self, timestep: Timestep, macro_out: bool) -> list[tuple[str, list[ps.Assignment]]]:
        """(name, assignments) of every pass of one parity, in call order."""
        stencil, method, dtype = self.stencil, self.method, self.dtype
        cqc = method.conserved_quantity_computation
        if not (cqc.zero_centered_pdfs and cqc.compressible):
            raise ValueError("SplitAA needs a compressible method with zero-centered PDFs")
        q, d = stencil.Q, stencil.D
        f = ps.Field.create_generic("f", d, index_shape=(q,), layout="fzyx", dtype=dtype)
        acc = ps.Field.create_generic("acc", d, index_shape=(d + 1,), layout="fzyx", dtype=dtype)
        scr = ps.Field.create_generic("scr", d, index_shape=(d + 2,), layout="fzyx", dtype=dtype)
        rho_out, u_out = ps.fields(f"rho, u({d}): {dtype}[{d}D]", layout="fzyx")
        accessor = get_accessor("aa", timestep)
        reads, writes = accessor.read(f, stencil), accessor.write(f, stencil)
        v = sp.symbols(f"v_:{q}")

        rules = []
        #   1. moments: delta_rho = sum f_i, j = sum c_i f_i
        for g0 in range(0, q, self.gm):
            group = range(g0, min(g0 + self.gm, q))
            loads = [ps.Assignment(v[i], reads[i]) for i in group]
            sums = [sp.Add(*[v[i] for i in group])] + [
                sp.Add(*[stencil[i][k] * v[i] for i in group]) for k in range(d)
            ]
            accum = [ps.Assignment(acc(k), (acc(k) if g0 else 0) + s) for k, s in enumerate(sums)]
            rules.append((f"m{g0 // self.gm}", loads + accum))

        #   2. macroscopic values
        drho, rho = cqc.density_deviation_symbol, cqc.density_symbol
        u = cqc.velocity_symbols
        macro = [ps.Assignment(drho, acc(0)), ps.Assignment(rho, drho + cqc.background_density)]
        macro += [ps.Assignment(u[k], acc(k + 1) / rho) for k in range(d)]
        macro += [ps.Assignment(scr(0), drho), ps.Assignment(scr(1), rho)]
        macro += [ps.Assignment(scr(k + 2), u[k]) for k in range(d)]
        if macro_out:
            macro += [ps.Assignment(rho_out(), rho)] + [ps.Assignment(u_out(k), u[k]) for k in range(d)]
        rules.append(("macro" if macro_out else "macro_noout", macro))

        #   3. collision of opposite-direction pairs; the rest direction goes with the first pass.
        #   lbmpy's equilibrium has the form feq_i = w_i (delta_rho + rho t_i(u)) (checked below). For a
        #   pair (i, j), t_i = e + o and t_j = e - o with the even part e = (t_i + t_j) / 2 and the odd
        #   part o = (t_i - t_j) / 2, so a pair costs one e and one o.
        eq, w = method.get_equilibrium_terms(), method.weights
        inv = [stencil.index(tuple(-c for c in stencil[i])) for i in range(q)]
        pairs = [(i, inv[i]) for i in range(q) if i < inv[i]]
        rest = [i for i in range(q) if inv[i] == i]
        t = [sp.expand((eq[i] / w[i] - drho) / rho) for i in range(q)]
        for i in range(q):
            if sp.expand(w[i] * (drho + rho * t[i]) - eq[i]) != 0:
                raise ValueError("SplitAA: equilibrium is not of the form w_i (delta_rho + rho t_i(u))")
        pre = [ps.Assignment(drho, scr(0)), ps.Assignment(rho, scr(1))]
        pre += [ps.Assignment(u[k], scr(k + 2)) for k in range(d)]
        for g0 in range(0, len(pairs), self.gp):
            group = pairs[g0 : g0 + self.gp]
            dirs = (rest if g0 == 0 else []) + [i for p in group for i in p]
            loads = [ps.Assignment(v[i], reads[i]) for i in dirs]   # every load before any store
            temps, ts = [], {i: t[i] for i in rest}
            for i, j in group:
                e, o = sp.Symbol(f"e_{i}"), sp.Symbol(f"o_{i}")
                temps += [ps.Assignment(e, sp.expand((t[i] + t[j]) / 2)), ps.Assignment(o, sp.expand((t[i] - t[j]) / 2))]
                ts[i], ts[j] = e + o, e - o
            stores = [
                ps.Assignment(writes[i], v[i] + self.omega * (w[i] * (drho + rho * ts[i]) - v[i])) for i in dirs
            ]
            #   common products (u_0**2, ..., rho w_i) once per pass; the stores stay last
            body = ps.simp.sympy_cse_on_assignment_list(temps + stores)
            rules.append((f"c{g0 // self.gp}", pre + loads + body))
        return rules

    def kernels(self, outputs=(True, False)):
        """{parity: {output: [(pass name, pystencils kernel)] in call order}}.

        The flavours with and without output differ in the macroscopic pass only; they share the
        kernel objects of the other passes.
        """
        result = {}
        for parity, ts in ((0, Timestep.EVEN), (1, Timestep.ODD)):
            cache, flavours = {}, {}
            for out in outputs:
                seq = []
                for name, rule in self.stage_rules(ts, out):
                    if name not in cache:
                        cfg = copy.deepcopy(self.gen_config)
                        cfg.function_name = f"{name}_{parity}"
                        cache[name] = ps.create_kernel(rule, cfg)
                    seq.append((name, cache[name]))
                flavours[out] = seq
            result[parity] = flavours
        return result

    @staticmethod
    def unique_kernels(kernels) -> list:
        """Every kernel of `kernels()` once."""
        seen, out = set(), []
        for flavours in kernels.values():
            for seq in flavours.values():
                for name, k in seq:
                    if k.name not in seen:
                        seen.add(k.name)
                        out.append(k)
        return out

    #   Driver variables for the passes' parameters (see driver_code)
    _ARGS = {
        "_data_f": "fb", "_data_acc": "acc", "_data_scr": "scr", "_data_rho": "rhob", "_data_u": "ub",
        "_stride_f_1": "sf1", "_stride_f_2": "sf2", "_stride_f_3": "sf3",
        "_stride_acc_1": "n0", "_stride_acc_2": "sc", "_stride_acc_3": "sc",
        "_stride_scr_1": "n0", "_stride_scr_2": "sc", "_stride_scr_3": "sc",
        "_stride_rho_1": "sr1", "_stride_rho_2": "sr2",
        "_stride_u_1": "su1", "_stride_u_2": "su2", "_stride_u_3": "su3",
    }

    def _call(self, kernel) -> str:
        args = []
        for p in kernel.parameters:
            size = re.fullmatch(r"_size_\w+_([012])", p.name)
            if size:
                args.append(("n0", "rows", "1")[int(size[1])])
            elif p.name in self._ARGS:
                args.append(self._ARGS[p.name])
            elif p.name == str(self.omega):
                args.append("omega")
            else:
                raise ValueError(f"SplitAA: unexpected parameter {p.name} of {kernel.name}")
        return f"{kernel.name}({', '.join(args)});"

    def driver_code(self, func_name: str, ctype: str, kernels=None) -> tuple[str, str]:
        """(declaration, definition) of the driver:

        void func_name(T* f, T* rho, T* u, int64_t n0, int64_t n1, int64_t n2, int64_t sf1, int64_t sf2,
                       int64_t sf3, int64_t sr1, int64_t sr2, int64_t su1, int64_t su2, int64_t su3,
                       bool odd, bool output, T omega)

        f, rho, u point at cell (0, 0, 0) of the block's interior; strides are in elements, x has stride 1.
        """
        kernels = kernels if kernels is not None else self.kernels()
        sig = (
            f"void {func_name}({ctype} * f, {ctype} * rho, {ctype} * u, int64_t n0, int64_t n1, int64_t n2, "
            "int64_t sf1, int64_t sf2, int64_t sf3, int64_t sr1, int64_t sr2, int64_t su1, int64_t su2, "
            f"int64_t su3, bool odd, bool output, {ctype} omega)"
        )

        def calls(parity, output):
            return "\n".join("            " + self._call(k) for _, k in kernels[parity][output])

        definition = f"""{sig}
{{
   //  Scratch of one block: 4 accumulators and 5 macroscopic values per cell, components 64-byte aligned
   static std::vector< {ctype} > storage;
   const int64_t sc{{ (int64_t({self.by}) * n0 + 15) / 16 * 16 }};
   if (storage.size() < size_t(9 * sc + 16)) {{ storage.resize(size_t(9 * sc + 16)); }}
   {ctype} * const acc{{ reinterpret_cast< {ctype} * >((reinterpret_cast< uintptr_t >(storage.data()) + 63) & ~uintptr_t(63)) }};
   {ctype} * const scr{{ acc + 4 * sc }};
   for (int64_t z = 0; z < n2; ++z)
      for (int64_t y0 = 0; y0 < n1; y0 += {self.by})
      {{
         const int64_t rows{{ n1 - y0 < {self.by} ? n1 - y0 : {self.by} }};
         {ctype} * const fb{{ f + y0 * sf1 + z * sf2 }};
         {ctype} * const rhob{{ rho + y0 * sr1 + z * sr2 }};
         {ctype} * const ub{{ u + y0 * su1 + z * su2 }};
         (void) rhob; (void) ub;
         if (!odd && output)
         {{
{calls(0, True)}
         }}
         else if (!odd)
         {{
{calls(0, False)}
         }}
         else if (output)
         {{
{calls(1, True)}
         }}
         else
         {{
{calls(1, False)}
         }}
      }}
}}
"""
        return sig + ";", definition
