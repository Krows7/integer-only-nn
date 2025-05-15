# Tools
## Build Flags

LINEAR_METRICS -- allow to print linear methods' calls metrics. If NES flag is specified, then requires to specifiy DEBUG_LOG_LEVEL > 0
NO_PRINT -- do not use stdio print methods (just ignore them)
NES -- the target is NES
EXIT_ON_ERROR -- if the error is occured, the program exits with code 1

## NES Binary Analysis

*Suggestions were taken from: https://llvm-mos.org/wiki/Current_status*
*Fitting banking from: https://github.com/llvm-mos/llvm-mos-sdk/blob/main/mos-platform/nes/ines.h*
*NES Target: https://llvm-mos.org/wiki/NES_targets*
*MMC3 Mapper: https://www.nesdev.org/wiki/MMC3*
*MMC1 Mapper: https://www.nesdev.org/wiki/MMC1*
*MMC5 Mapper: https://www.nesdev.org/wiki/MMC5*

LLVM MOS creates binaries and intermediates in llvm-build/ folder

To analyze object files, inspect llvm-build/project.map

Or, use:

To check full .elf file:
```
llvm-readelf -a llvm-build/project.nes.elf
```

To inspect sections sizes:
```
llvm-objdump --section-headers llvm-build/project.nes.elf
```

Use can also check size of all symbols:
```
llvm-objdump --all-headers llvm-build/project.nes.elf
```

To disassemble binary with debugging symbols:
```
llvm-objdump -d --start-address=0x0 llvm-build/project.nes.elf
```