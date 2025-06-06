#!/bin/bash

mkdir -p llvm-build

# Gonna compile specific tests/ file

# mos-nes-mmc3-clang -g -Oz -Wl,-Map=llvm-build/project.map -o llvm-build/project.nes -Isrc/api -Isrc/nes src/api/*.c src/nes/*.cc src/nes/*.c src/tests/nes/iris-nes.c -DNES -lneslib

# mos-nes-mmc3-clang -g -Oz -Wl,-Map=llvm-build/project.map -o llvm-build/project.nes -Isrc/api -Isrc/nes chr/*.s src/api/*.c src/nes/*.cc src/nes/*.c src/tests/nes/iris-nes.c -DNES -lneslib
# mos-nes-mmc3-clang -g -O1 -Wl,-Map=llvm-build/project.map -o llvm-build/project.nes -Isrc/api -Isrc/nes chr/*.s src/api/*.c src/nes/*.cc src/nes/*.c src/tests/nes/test-ppu.c -DNES -lneslib

# mos-nes-mmc3-clang -g -O1 -Wl,-Map=llvm-build/project.map -o llvm-build/project.nes -Isrc/api -Isrc/nes chr/*.s src/api/*.c src/nes/*.cc src/nes/*.c src/tests/nes/game.c -DNES -lneslib
mos-nes-mmc3-clang -g -Oz -Wl,-Map=llvm-build/project.map -o llvm-build/project.nes -Isrc/api -Isrc/nes chr/*.s src/api/*.c src/nes/*.cc src/nes/*.c src/tests/nes/iris-with-game.c -DNES -lneslib
# mos-nes-mmc3-clang -g -Oz -Wl,-Map=llvm-build/project.map -o llvm-build/project.nes -Isrc/api -Isrc/nes chr/*.s src/api/*.c src/nes/*.cc src/nes/*.c src/tests/nes/iris-nes.c -DNES -lneslib