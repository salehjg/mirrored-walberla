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
(`LBMOptimisation(cse_global=True)`).

- sweepgen never enables pystencils vectorization by itself, so the explicit variant enables
  it in `SweepgenLbmCPU.py`.
- The explicit variant also sets `assume_inner_stride_one`. waLBerla fields use the fzyx
  layout, so x is the fastest index with stride 1.

## Run

```
SweepgenLbmCPU_<variant>[_cse] [cells per block and direction = 64] [timesteps = 200] [warmup = 10]
```

The domain has one block per MPI process. Each run prints one line:

```
SweepgenLbmCPU variant=<executable> real_t=<bits> cells=<n> steps=<n> seconds=<t> mlups=<x> energy=<e>
```

`energy` is the total kinetic energy after the run, summed in double precision. All variants
of one build must agree on it up to rounding.

The precision follows the waLBerla build (`WALBERLA_DOUBLE_ACCURACY`). The board scripts
that build and run the whole matrix are in the outer repository:

- `scripts/walberla_lbm_bench.sh`: one build dir, all six variants on one core
- `scripts/walberla_bench_all.sh`: the GCC/Clang × f64/f32 matrix
- `scripts/walberla_bench_followup.sh`: reruns after failed builds, then the multi-core runs
- `scripts/walberla_bench_mpi.sh`: the multi-core runs (4 and 8 MPI processes)
