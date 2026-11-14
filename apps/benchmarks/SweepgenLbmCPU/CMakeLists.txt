#  SweepgenLbmCPU: throughput of the DoubleShearLayer LBM sweeps, built once per code
#  generation variant so the variants can be compared on the same machine:
#
#    scalar    GenericCPU sweeps, no SIMD at all: -march=rv64gc on RISC-V (no V
#              extension), -fno-tree-vectorize elsewhere
#    autovec   GenericCPU sweeps with waLBerla's flags: the compiler may auto-vectorize
#    explicit  pystencils' vector backend for the host (Target.CurrentCPU, vectorized):
#              RVV intrinsics on the SpacemiT K1, AVX on x86
#
#  Each variant is also built with lbmpy's global CSE (suffix _cse; see SweepgenLbmCPU.py).
#  Further variants (docs/notes3):
#    _pf   explicit with software prefetching, 128 bytes ahead
#    _aa   in-place AA streaming instead of pull: one PDF field, even/odd sweeps
#  Run: SweepgenLbmCPU_<variant>[_cse][_aa][_pf] [cells per direction] [timesteps] [warmup]

#  sweepgen_lbm_variant( <target> <scalar|autovec|explicit> <SweepgenLbmCPU.py arguments...> )
function( sweepgen_lbm_variant _target _family )
   add_executable( ${_target} SweepgenLbmCPU.cpp )
   walberla_generate_sources( ${_target} SCRIPTS SweepgenLbmCPU.py SCRIPT_ARGS --variant=${_family} ${ARGN} )
   target_link_libraries( ${_target} PRIVATE walberla::v8 )
   target_compile_definitions( ${_target} PRIVATE SWEEPGEN_LBM_VARIANT="${_target}" )
   if( "--streaming-pattern=aa" IN_LIST ARGN )
      target_compile_definitions( ${_target} PRIVATE SWEEPGEN_LBM_AA )
   endif()
   foreach( _arg ${ARGN} )
      if( _arg MATCHES "^--split=" )
         target_compile_definitions( ${_target} PRIVATE SWEEPGEN_LBM_SPLIT )
      endif()
   endforeach()

   if( _family STREQUAL "scalar" )
      #  Added after waLBerla's own flags, so this -march wins
      if( CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv64" )
         target_compile_options( ${_target} PRIVATE -march=rv64gc )
      else()
         target_compile_options( ${_target} PRIVATE -fno-tree-vectorize )
      endif()
   endif()
endfunction()

foreach( _family scalar autovec explicit )
   sweepgen_lbm_variant( SweepgenLbmCPU_${_family} ${_family} )
   sweepgen_lbm_variant( SweepgenLbmCPU_${_family}_cse ${_family} --cse-global )
endforeach()

sweepgen_lbm_variant( SweepgenLbmCPU_explicit_pf explicit --prefetch-distance=128 )
sweepgen_lbm_variant( SweepgenLbmCPU_explicit_cse_pf explicit --cse-global --prefetch-distance=128 )

sweepgen_lbm_variant( SweepgenLbmCPU_scalar_cse_aa scalar --cse-global --streaming-pattern=aa )
sweepgen_lbm_variant( SweepgenLbmCPU_explicit_cse_aa explicit --cse-global --streaming-pattern=aa )
sweepgen_lbm_variant( SweepgenLbmCPU_explicit_cse_aa_pf explicit --cse-global --streaming-pattern=aa
                      --prefetch-distance=128 )

#  Direction-split AA stream/collide (lbm_split.py): rows per block, directions per moment pass and
#  opposite pairs per collision pass. A block row of doubles takes twice the scratch of floats, so
#  double precision uses one-row blocks (docs/notes3 Loops 7, 14).
if( WALBERLA_DOUBLE_ACCURACY )
   set( _split 1,5,3 )
else()
   set( _split 2,5,3 )
endif()
sweepgen_lbm_variant( SweepgenLbmCPU_explicit_cse_aa_split explicit --cse-global --streaming-pattern=aa
                      --split=${_split} )
