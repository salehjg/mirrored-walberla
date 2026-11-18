#  SweepgenLdcCPU: throughput of the SweepgenLbmCPU LBM sweeps (D3Q19, SRT, compressible) in a
#  lid-driven cavity instead of a periodic box. The walls are the setup of waLBerla's
#  UniformGridCPU LDC scenario: the north face (y max) is a lid moving with (u_lid, 0, 0), the
#  other five faces are no-slip walls. One executable per code generation variant, as in
#  SweepgenLbmCPU (see its CMakeLists.txt for the variant families):
#
#    scalar    GenericCPU sweeps, no SIMD at all: -march=rv64gc on RISC-V (no V extension),
#              -fno-tree-vectorize elsewhere
#    autovec   GenericCPU sweeps with waLBerla's flags: the compiler may auto-vectorize
#    explicit  pystencils' vector backend for the host (Target.CurrentCPU, vectorized):
#              RVV intrinsics on the SpacemiT K1, AVX on x86
#
#  Suffixes: _cse global CSE, _pf software prefetch 128 bytes ahead, _aa in-place AA streaming,
#  _split the direction-split AA stream/collide of ../SweepgenLbmCPU/lbm_split.py.
#  The wall sweeps (sparse NoSlip and UBB) are the same generated code in every variant.
#  Run: SweepgenLdcCPU_<variant> [cells per block and direction] [timesteps] [warmup]

#  sweepgen_ldc_variant( <target> <scalar|autovec|explicit> <SweepgenLdcCPU.py arguments...> )
function( sweepgen_ldc_variant _target _family )
   add_executable( ${_target} SweepgenLdcCPU.cpp )
   walberla_generate_sources( ${_target} SCRIPTS SweepgenLdcCPU.py
                              DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/ldc_boundaries.py
                                      ${CMAKE_CURRENT_SOURCE_DIR}/../SweepgenLbmCPU/lbm_split.py
                              SCRIPT_ARGS --variant=${_family} ${ARGN} )
   target_link_libraries( ${_target} PRIVATE walberla::v8 )
   target_compile_definitions( ${_target} PRIVATE SWEEPGEN_LDC_VARIANT="${_target}"
                               SWEEPGEN_LDC_COMPILER="${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}" )
   if( "--streaming-pattern=aa" IN_LIST ARGN )
      target_compile_definitions( ${_target} PRIVATE SWEEPGEN_LDC_AA )
   endif()
   foreach( _arg ${ARGN} )
      if( _arg MATCHES "^--split=" )
         target_compile_definitions( ${_target} PRIVATE SWEEPGEN_LDC_SPLIT )
      endif()
   endforeach()

   if( _family STREQUAL "scalar" )
      #  Added after waLBerla's own flags, so this -march wins for the app's own sources. The
      #  presentation4 scalar build trees also compile all of waLBerla with -march=rv64gc.
      if( CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv64" )
         target_compile_options( ${_target} PRIVATE -march=rv64gc )
      else()
         target_compile_options( ${_target} PRIVATE -fno-tree-vectorize )
      endif()
   endif()
endfunction()

#  The 11 variants of docs/presentation3 (scripts/presentation3/{scalar,autovec,rvv}.sh)
sweepgen_ldc_variant( SweepgenLdcCPU_scalar scalar )
sweepgen_ldc_variant( SweepgenLdcCPU_scalar_cse scalar --cse-global )
sweepgen_ldc_variant( SweepgenLdcCPU_scalar_cse_aa scalar --cse-global --streaming-pattern=aa )

sweepgen_ldc_variant( SweepgenLdcCPU_autovec autovec )
sweepgen_ldc_variant( SweepgenLdcCPU_autovec_cse autovec --cse-global )
sweepgen_ldc_variant( SweepgenLdcCPU_autovec_cse_aa autovec --cse-global --streaming-pattern=aa )

sweepgen_ldc_variant( SweepgenLdcCPU_explicit_cse explicit --cse-global )
sweepgen_ldc_variant( SweepgenLdcCPU_explicit_cse_pf explicit --cse-global --prefetch-distance=128 )
sweepgen_ldc_variant( SweepgenLdcCPU_explicit_cse_aa explicit --cse-global --streaming-pattern=aa )
sweepgen_ldc_variant( SweepgenLdcCPU_explicit_cse_aa_pf explicit --cse-global --streaming-pattern=aa
                      --prefetch-distance=128 )

#  Direction-split AA stream/collide: the block sizes of SweepgenLbmCPU (docs/notes3 Loops 7, 14)
if( WALBERLA_DOUBLE_ACCURACY )
   set( _split 1,5,3 )
else()
   set( _split 2,5,3 )
endif()
sweepgen_ldc_variant( SweepgenLdcCPU_explicit_cse_aa_split explicit --cse-global --streaming-pattern=aa
                      --split=${_split} )
