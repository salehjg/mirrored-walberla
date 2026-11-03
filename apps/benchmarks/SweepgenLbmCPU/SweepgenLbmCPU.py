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

"""Code generator of the SweepgenLbmCPU benchmark (see CMakeLists.txt).

The LBM sweeps are those of the DoubleShearLayer example: D3Q19, SRT, compressible,
relaxation rate `omega`. They are generated in one of three variants that are built as
separate executables and compared on the same machine:

  scalar, autovec  Target.GenericCPU: plain C++ loops. The two variants differ only in
                   their compiler flags (see CMakeLists.txt).
  explicit         Target.CurrentCPU with the pystencils vectorizer enabled: explicit SIMD
                   intrinsics for the host, i.e. RVV on the SpacemiT K1 and AVX on x86.
                   sweepgen never enables vectorization by itself, so it is requested here,
                   as a user of the AVX backend would.

--cse-global also runs a global common-subexpression elimination over the update rule
(lbmpy's `cse_global`). Without it the sweep repeats many subterms. Compilers remove the
repeats from scalar code, but GCC does not remove repeated RVV intrinsic calls.
"""

import argparse

import sympy as sp
import pystencils as ps

from lbmpy import LBStencil, Stencil, LBMConfig, LBMOptimisation, Method

from pystencilssfg import SourceFileGenerator
from sweepgen import Sweep
from sweepgen.build_config import DEBUG
from sweepgen.symbolic import cell
from sweepgen.prefabs import LbmBulk

#   Only used when the script runs outside the waLBerla build system
DEBUG.use_cpu_default()

with SourceFileGenerator(keep_unknown_argv=True) as sfg:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("scalar", "autovec", "explicit"), required=True)
    parser.add_argument("--cse-global", action="store_true")
    #   For generating the explicit variant on another machine (e.g. RISCV_RVV on x86)
    parser.add_argument("--explicit-target", default="CurrentCPU")
    args = parser.parse_args(sfg.context.argv)

    Sweep.use_v8core_fields()
    sfg.namespace("SweepgenLbmCPU::gen")

    #   Code generator config of the timed LBM sweeps
    if args.variant == "explicit":
        lbm_gen_config = ps.CreateKernelConfig(target=ps.Target[args.explicit_target])
        lbm_gen_config.cpu.vectorize.enable = True
        #   waLBerla fields are fzyx with x fastest, so the innermost loop has stride 1:
        #   the same contract the AVX backend relies on
        lbm_gen_config.cpu.vectorize.assume_inner_stride_one = True
    else:
        lbm_gen_config = ps.CreateKernelConfig(target=ps.Target.GenericCPU)

    stencil = LBStencil(Stencil.D3Q19)
    lbm_config = LBMConfig(
        stencil=stencil,
        method=Method.SRT,
        compressible=True,
        relaxation_rate=sp.Symbol("omega"),
    )
    #   LbmBulk's own default, plus the optional global CSE
    lbm_optimisation = LBMOptimisation(
        simplification="auto",
        cse_global=args.cse_global,
        field_layout="fzyx",
    )

    lbm_bulk = LbmBulk(sfg, "LBM", lbm_config, lbm_optimisation, lbm_gen_config)
    sfg.generate(lbm_bulk)

    #   Initial state of the DoubleShearLayer example. It is not timed, so it is always
    #   generated as plain code (the vectorizers do not support tanh or Piecewise).
    rho, u = lbm_bulk.rho, lbm_bulk.u
    u_0, kappa, delta = sp.symbols("u_0, kappa, delta")
    initial_state_assignments = [
        ps.Assignment(rho(), 1),
        ps.Assignment(
            u(0),
            sp.Piecewise(
                (u_0 * sp.tanh(kappa * (cell.y() - sp.Rational(1, 4))), cell.y() <= sp.Rational(1, 2)),
                (u_0 * sp.tanh(kappa * (sp.Rational(3, 4) - cell.y())), cell.y() > sp.Rational(1, 2)),
            ),
        ),
        ps.Assignment(u(1), delta * u_0 * sp.sin(2 * sp.pi * (cell.x() + sp.Rational(1, 4)))),
        ps.Assignment(u(2), 0),
    ]
    sfg.generate(
        Sweep(
            "SetInitialState",
            initial_state_assignments,
            ps.CreateKernelConfig(target=ps.Target.GenericCPU),
        )
    )
