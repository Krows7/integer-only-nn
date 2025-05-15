#!/bin/bash

# Gonna compile specific tests/ file
mos-nes-mmc3-clang -g -Oz -Wl,-Map=llvm-build/project.map -o llvm-build/project.nes -Isrc/api -Isrc/nes src/api/*.c src/nes/*.cc src/nes/*.c src/tests/nes/iris-nes.c -DNES -lneslib