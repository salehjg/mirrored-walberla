# SweepgenLbmCPU

The benchmark measures the throughput (MLUP/s) of sweepgen-generated lattice Boltzmann sweeps
in each code generation variant. The sweeps are those of the `DoubleShearLayer` example
(D3Q19, SRT, compressible), run in a fully periodic domain. The timed loop is
stream-collide plus halo exchange.

## Variants

The CMake file builds one executable per variant:

| executable | sweeps | compiler flags |
|---|---|---|
| `SweepgenLbmCPU_scalar` | `Target.GenericCPU` (plain C++) | `-march=rv64gc` on RISC-V (no V extension), `-fno-tree-vectorize` elsewhere |
| `SweepgenLbmCPU_autovec` | `Target.GenericCPU` (plain C++) | waLBerla's flags: the compiler may auto-vectorize |
| `SweepgenLbmCPU_explicit` | `Target.CurrentCPU`, pystencils vectorizer on | explicit SIMD intrinsics: RVV on RISC-V, AVX on x86 |

Each variant is also built as `*_cse`, with lbmpy's global common-subexpression elimination
(`LBMOptimisation(cse_global=True)`). Further variants:

| executable | difference |
|---|---|
| `SweepgenLbmCPU_explicit[_cse]_pf` | software prefetching 128 bytes ahead (`cpu.vectorize.prefetch_distance`, RVV only) |
| `SweepgenLbmCPU_{scalar,autovec}_cse_aa`, `SweepgenLbmCPU_explicit_cse_aa[_pf]` | in-place AA streaming instead of pull (below) |
| `SweepgenLbmCPU_explicit_cse_aa_split` | AA streaming with the direction-split stream/collide of `lbm_split.py` (below) |

With `--streaming-pattern=aa` (`SWEEPGEN_LBM_AA` in the C++ file), sweepgen's `LbmBulk`
generates `StreamCollideEven` and `StreamCollideOdd` on one PDF field instead of
`StreamCollide` on two. The even sweep reads and writes each cell's own PDFs. The odd sweep reads
from and writes to the neighbors, so the ghost layers are exchanged with
`halo_exchange::aaPullSync` after even sweeps and `halo_exchange::aaPushSync` after odd sweeps.
The AA sweep moves about 30% fewer bytes per cell and has 23 instead of 42 memory streams.

With `--split=BY,GM,GP` (explicit variant with the AA pattern; `SWEEPGEN_LBM_SPLIT` in the C++
file), `SweepgenLbmCPU.py` also generates the direction-split stream/collide of `lbm_split.py`
and the app runs it instead of `StreamCollideEven`/`StreamCollideOdd`. It works on blocks of BY
rows of one z-plane: moments in passes of GM directions into a scratch, the macroscopic values,
then the collision in passes of GP opposite-direction pairs, so that only a few memory streams
are live at a time. Every pass is a pystencils kernel built from lbmpy's method (SRT only). The
CMake file uses `2,5,3` in single and `1,5,3` in double precision.

Every variant also contains stream/collide sweeps without the density/velocity output
(`LbmBulk(..., output_free_sweeps=True)`): 4 fewer store streams per cell.

- sweepgen never enables pystencils vectorization by itself, so the explicit variant enables
  it in `SweepgenLbmCPU.py`.
- The explicit variant also sets `assume_inner_stride_one`. waLBerla fields use the fzyx
  layout, so x is the fastest index with stride 1.

## Run

```
[SWEEPGEN_LBM_OUTPUT=last] SweepgenLbmCPU_<variant>[_cse][_aa][_pf|_split] [cells per block and direction = 64] [timesteps = 200] [warmup = 10]
```

With `SWEEPGEN_LBM_OUTPUT=last`, only the last time step writes the density and velocity fields
(the checksum reads the velocity after the loop); all other steps use the output-free sweeps.
The default, `every`, writes them in every step.

The domain has one block per MPI process. Each run prints one line:

```
SweepgenLbmCPU variant=<executable> real_t=<bits> cells=<n> steps=<n> output=<every|last> seconds=<t> mlups=<x> energy=<e>
```

`energy` is the total kinetic energy after the run, summed in double precision. All variants
of one build, pull and AA, must agree on it up to rounding, for any number of processes with the
same total domain.

The precision follows the waLBerla build (`WALBERLA_DOUBLE_ACCURACY`). The board scripts
that build and run the whole matrix are in the outer repository:

- `scripts/walberla_lbm_bench.sh`: one build dir, all variants on one core
- `scripts/walberla_bench_all.sh`: the GCC/Clang × f64/f32 matrix
- `scripts/walberla_bench_followup.sh`: reruns after failed builds, then the multi-core runs
- `scripts/walberla_bench_mpi.sh`: the multi-core runs (4 and 8 MPI processes)
- `scripts/walberla_bench_{scalar,autovec,rvv}.sh <f32|f64> <cores>`: one experiment (GCC), on one
  core or on 4 or 8 cores with MPI; `scripts/walberla_bench_presentation.sh` runs all of them
