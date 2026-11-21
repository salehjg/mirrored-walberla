# Vector register width (VLEN) of a RISC-V build machine with the V extension, in bits.
#
# pystencils generates vector-length-specific RVV code (Target.RISCV_RVV128 or RISCV_RVV256), so the
# code generator and the compiler flags (-march=rv64gcv_zvl<VLEN>b) must both know it. Linux does not
# report VLEN in /proc/cpuinfo, so a tiny program reads the `vlenb` register (CSR 0xc22, given by
# number so that no -march flag is needed). Included from GNU.cmake and Clang.cmake on riscv64 hosts
# whose ISA string has the V extension. Set WALBERLA_RISCV_VLEN by hand for cross builds.
include_guard(GLOBAL)

set(WALBERLA_RISCV_VLEN "" CACHE STRING
    "RISC-V vector register width in bits (128 or 256); measured on the build machine if empty")

if(NOT WALBERLA_RISCV_VLEN)
  try_run(_riscv_vlen_run _riscv_vlen_compiled
          SOURCE_FROM_CONTENT riscv_vlen_probe.cpp [=[
#include <cstdio>
int main()
{
    unsigned long vlenb;
    __asm__ volatile("csrr %0, 0xc22" : "=r"(vlenb));
    std::printf("%lu", vlenb * 8);
    return 0;
}
]=]
          RUN_OUTPUT_VARIABLE _riscv_vlen_out)
  if(NOT _riscv_vlen_compiled OR NOT _riscv_vlen_run EQUAL 0)
    message(FATAL_ERROR "Could not measure the RISC-V vector length: set WALBERLA_RISCV_VLEN (128 or 256)")
  endif()
  set(WALBERLA_RISCV_VLEN "${_riscv_vlen_out}" CACHE STRING
      "RISC-V vector register width in bits (128 or 256); measured on the build machine if empty" FORCE)
  message(STATUS "RISC-V VLEN: ${WALBERLA_RISCV_VLEN} bits (measured)")
else()
  message(STATUS "RISC-V VLEN: ${WALBERLA_RISCV_VLEN} bits (set by hand)")
endif()

if(NOT WALBERLA_RISCV_VLEN MATCHES "^(128|256)$")
  message(FATAL_ERROR "WALBERLA_RISCV_VLEN=${WALBERLA_RISCV_VLEN}: pystencils supports 128 and 256")
endif()
