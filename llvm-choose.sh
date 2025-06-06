#!/bin/bash

mkdir -p llvm-build

mos-nes-mmc3-clang -g -Oz -Wl,-Map=llvm-build/project.map -o llvm-build/project.nes -Isrc/api -Isrc/nes -Isrc/game src/game/*.c chr/*.s src/api/*.c src/nes/*.cc src/nes/*.c src/tests/nes/$1.c -DNES -lneslib
# mos-nes-mmc3-clang -g -Oz -Wl,-Map=llvm-build/project.map -o llvm-build/project.nes -Isrc/api -Isrc/nes chr/*.s src/api/*.c src/nes/*.cc src/nes/*.c src/tests/nes/$1.c -DNES -lneslib