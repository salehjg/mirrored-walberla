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

"""Code generator of the SweepgenLdcCPU benchmark (see CMakeLists.txt and README.md).

The bulk LBM sweeps are generated exactly as in ../SweepgenLbmCPU/SweepgenLbmCPU.py: D3Q19, SRT,
compressible, relaxation rate `omega`, in the variants scalar / autovec / explicit, with the same
options --cse-global, --prefetch-distance, --rvv-lmul, --streaming-pattern and --split (the split module is
../SweepgenLbmCPU/lbm_split.py itself, not a copy). See that file for what each option does.

The lid-driven cavity adds two sparse boundary sweeps (sweepgen `SparseBoundary` with lbmpy's
boundary conditions):

  NoSlip  half-way bounce-back, on the links into the W, E, S, B and T walls
  UBB     velocity bounce-back with the lid velocity (u_lid, 0, 0), on the links into the N wall

With --streaming-pattern aa each comes twice, for the state after an even and after an odd time
step (NoSlipEven/NoSlipOdd, UBBEven/UBBOdd; ldc_boundaries.py). The boundary sweeps are generated
for Target.GenericCPU in every variant, so their code is the same in all executables and only the
compiler flags differ. Their data type follows the build precision.
"""

import argparse
import pathlib
import sys

import sympy as sp
import pystencils as ps

from lbmpy import LBStencil, Stencil, LBMConfig, LBMOptimisation, Method
from lbmpy.advanced_streaming import Timestep
from lbmpy.boundaries import NoSlip, UBB

from pystencilssfg import SourceFileGenerator
from sweepgen import Sweep
from sweepgen.boundaries import SparseBoundary
from sweepgen.build_config import DEBUG, get_build_config
from sweepgen.prefabs import LbmBulk

from ldc_boundaries import AASparseBoundary

#   Only used when the script runs outside the waLBerla build system
DEBUG.use_cpu_default()

with SourceFileGenerator(keep_unknown_argv=True) as sfg:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("scalar", "autovec", "explicit"), required=True)
    parser.add_argument("--cse-global", action="store_true")
    parser.add_argument("--prefetch-distance", type=int, default=None)
    parser.add_argument("--streaming-pattern", choices=("pull", "aa"), default="pull")
    parser.add_argument("--split", default=None, help="BY,GM,GP: direction-split AA stream/collide")
    #   Default: the build's CPU target, which on RISC-V is the RVV target of the vector length CMake
    #   measured. Set it to generate the explicit variant for another machine (e.g. RISCV_RVV128 on x86).
    parser.add_argument("--explicit-target", default=None)
    parser.add_argument("--rvv-lmul", type=int, default=None, help="RVV register grouping: 1, 2 or 4")
    args = parser.parse_args(sfg.context.argv)

    Sweep.use_v8core_fields()
    sfg.namespace("SweepgenLdcCPU::gen")
    build_config = get_build_config(sfg)

    #   ---- Bulk sweeps: identical to SweepgenLbmCPU.py ----------------------------------------
    if args.variant == "explicit":
        target = ps.Target[args.explicit_target] if args.explicit_target else get_build_config(sfg).target
        lbm_gen_config = ps.CreateKernelConfig(target=target)
        lbm_gen_config.cpu.vectorize.enable = True
        #   waLBerla fields are fzyx with x fastest, so the innermost loop has stride 1
        lbm_gen_config.cpu.vectorize.assume_inner_stride_one = True
        if args.prefetch_distance:
            lbm_gen_config.cpu.vectorize.prefetch_distance = args.prefetch_distance
        if args.rvv_lmul:
            lbm_gen_config.cpu.vectorize.rvv_lmul = args.rvv_lmul
    elif args.prefetch_distance or args.rvv_lmul:
        raise ValueError("--prefetch-distance and --rvv-lmul need the explicit variant")
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
    lbm_optimisation = LBMOptimisation(
        simplification="auto",
        cse_global=args.cse_global,
        field_layout="fzyx",
    )

    lbm_bulk = LbmBulk(sfg, "LBM", lbm_config, lbm_optimisation, lbm_gen_config, output_free_sweeps=True)
    sfg.generate(lbm_bulk)

    if args.split:
        if args.variant != "explicit" or args.streaming_pattern != "aa":
            raise ValueError("--split needs the explicit variant and --streaming-pattern aa")
        #   The split module of SweepgenLbmCPU, imported from there
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "SweepgenLbmCPU"))
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

    #   ---- Walls of the lid-driven cavity ------------------------------------------------------
    #   Plain C++ in every variant (index-list loops), so all executables share the same wall code
    build_config.override.target = ps.Target.GenericCPU
    wall_dtype = "float64" if build_config.use_double_precision else "float32"
    u_lid = sp.Symbol("u_lid")

    def generate_walls(suffix: str, make_boundary):
        sfg.generate(make_boundary(NoSlip(name=f"NoSlip{suffix}")))
        sfg.generate(make_boundary(UBB((u_lid, 0, 0), name=f"UBB{suffix}", data_type=wall_dtype)))

    if args.streaming_pattern == "pull":
        generate_walls("", lambda bc: SparseBoundary(bc, lbm_bulk.lb_method, lbm_bulk.pdfs))
    else:
        for timestep, suffix in ((Timestep.EVEN, "Even"), (Timestep.ODD, "Odd")):
            generate_walls(
                suffix,
                lambda bc, ts=timestep: AASparseBoundary(bc, lbm_bulk.lb_method, lbm_bulk.pdfs, ts),
            )
