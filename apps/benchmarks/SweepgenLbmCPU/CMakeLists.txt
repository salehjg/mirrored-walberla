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
#  Run: SweepgenLbmCPU_<variant>[_cse] [cells per direction] [timesteps] [warmup]

foreach( _variant scalar autovec explicit )
   foreach( _cse default cse )
      set( _target SweepgenLbmCPU_${_variant} )
      set( _scriptArgs --variant=${_variant} )
      if( _cse STREQUAL "cse" )
         set( _target ${_target}_cse )
         list( APPEND _scriptArgs --cse-global )
      endif()

      add_executable( ${_target} SweepgenLbmCPU.cpp )
      walberla_generate_sources( ${_target} SCRIPTS SweepgenLbmCPU.py SCRIPT_ARGS ${_scriptArgs} )
      target_link_libraries( ${_target} PRIVATE walberla::v8 )
      target_compile_definitions( ${_target} PRIVATE SWEEPGEN_LBM_VARIANT="${_target}" )

      if( _variant STREQUAL "scalar" )
         #  Added after waLBerla's own flags, so this -march wins
         if( CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv64" )
            target_compile_options( ${_target} PRIVATE -march=rv64gc )
         else()
            target_compile_options( ${_target} PRIVATE -fno-tree-vectorize )
         endif()
      endif()
   endforeach()
endforeach()
