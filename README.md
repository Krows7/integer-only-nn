# C Implementation of Neural Network Framework

## Prerequisites
For running tests, download datasets using ```./download-data.sh```

Make sure you have GCC, CMake and valgrind installed.

For NES compiling, you need to install [LLVM-MOS-SDK](https://github.com/llvm-mos/llvm-mos-sdk) in nested folder with this project.

## Usage
For example, to build and run "digits-new.c" test:
```bash
make clean all debug BIN=digits-new
```
To analyze:
```bash
make clean all && valgrind --leak-check=full ./build/digits-new
```

To build NES Project:
```bash
./llvm-mos.sh
```

