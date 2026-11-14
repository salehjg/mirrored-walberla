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

--prefetch-distance N (explicit variant only) makes the vector loop prefetch every field
access N bytes ahead (`cpu.vectorize.prefetch_distance`). The D3Q19 sweep has 19 read and
23 write streams, more than the hardware prefetcher of the SpacemiT X60 follows; without
help every cache miss costs a full DRAM latency (docs/notes3 in the outer repository).

--streaming-pattern aa streams in place with lbmpy's AA pattern instead of pull: one PDF field
instead of two, alternating even and odd sweeps (StreamCollideEven/StreamCollideOdd), and
SweepgenLbmCPU.cpp must be compiled with SWEEPGEN_LBM_AA (see CMakeLists.txt). The sweep then
moves about 30% fewer bytes per cell and has 23 instead of 42 memory streams.

--split BY,GM,GP (explicit variant with the aa pattern) also generates the direction-split
stream/collide of lbm_split.py: blocks of BY rows, moments in passes of GM directions, collision
in passes of GP opposite-direction pairs, so that only a few memory streams are live at a time.
SweepgenLbmCPU.cpp must then be compiled with SWEEPGEN_LBM_SPLIT, and runs it instead of
StreamCollideEven/StreamCollideOdd.

Every variant also contains the stream/collide sweeps without density/velocity output
(LbmBulk's output_free_sweeps): 4 fewer store streams. SweepgenLbmCPU.cpp uses them for every
time step except the last one when run with SWEEPGEN_LBM_OUTPUT=last.
"""

import argparse

import sympy as sp
import pystencils as ps

from lbmpy import LBStencil, Stencil, LBMConfig, LBMOptimisation, Method

from pystencilssfg import SourceFileGenerator
from sweepgen import Sweep
from sweepgen.build_config import DEBUG, get_build_config
from sweepgen.symbolic import cell
from sweepgen.prefabs import LbmBulk

#   Only used when the script runs outside the waLBerla build system
DEBUG.use_cpu_default()

with SourceFileGenerator(keep_unknown_argv=True) as sfg:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("scalar", "autovec", "explicit"), required=True)
    parser.add_argument("--cse-global", action="store_true")
    parser.add_argument("--prefetch-distance", type=int, default=None)
    parser.add_argument("--streaming-pattern", choices=("pull", "aa"), default="pull")
    parser.add_argument("--split", default=None, help="BY,GM,GP: direction-split AA stream/collide")
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
        if args.prefetch_distance:
            lbm_gen_config.cpu.vectorize.prefetch_distance = args.prefetch_distance
    elif args.prefetch_distance:
        raise ValueError("--prefetch-distance needs the explicit variant")
    else:
        lbm_gen_config = ps.CreateKernelConfig(target=ps.Target.GenericCPU)

    stencil = LBStencil(Stencil.D3Q19)
    lbm_config = LBMConfig(
        stencil=stencil,
        method=Method.SRT,
        compressible=True,
        relaxation_rate=sp.Symbol("omega"),
        streaming_pattern=args.streaming_pattern,
    )
    #   LbmBulk's own default, plus the optional global CSE
    lbm_optimisation = LBMOptimisation(
        simplification="auto",
        cse_global=args.cse_global,
        field_layout="fzyx",
    )

    #   Also the sweeps without density/velocity output, for the time steps whose macroscopic
    #   values are not needed (SweepgenLbmCPU.cpp: SWEEPGEN_LBM_OUTPUT=last)
    lbm_bulk = LbmBulk(sfg, "LBM", lbm_config, lbm_optimisation, lbm_gen_config, output_free_sweeps=True)
    sfg.generate(lbm_bulk)

    if args.split:
        if args.variant != "explicit" or args.streaming_pattern != "aa":
            raise ValueError("--split needs the explicit variant and --streaming-pattern aa")
        #   lbm_split.py lives next to this script
        from lbm_split import SplitAA

        by, gm, gp = (int(x) for x in args.split.split(","))
        split_config = get_build_config(sfg).get_pystencils_config()
        split_config.override(lbm_gen_config)
        #   The driver passes each block's extent: no ghost layers to subtract
        split_config.ghost_layers = 0
        dtype = split_config.get_option("default_dtype")
        ctype = "double" if "64" in str(dtype) else "float"
        split = SplitAA(lbm_config, str(dtype), by, gm, gp, split_config)
        kernels = split.kernels()
        unique = SplitAA.unique_kernels(kernels)
        for header in sorted(set().union(*(k.required_headers for k in unique)) | {"<cstdint>", "<vector>"}):
            sfg.include(header.strip('"'))
        decl, definition = split.driver_code("streamCollide", ctype, kernels)
        with sfg.namespace("LBM_SPLIT"):
            sfg.code(
                "/** Direction-split AA stream/collide of a block (lbm_split.py): f, rho, u point at the block's",
                "    interior cell (0, 0, 0); strides in elements; odd: odd time step; output: write rho and u */",
                decl,
            )
            sfg.code("namespace {", *(k.get_c_code() for k in unique), "} // namespace", definition, impl=True)

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
