#!/bin/bash

# BIN_TEST=ext/digits
# BIN_TEST=nes/iris-nes

# # # Rejection Sampling
# # make clean all BASE_CFLAGS+=" -DRANDOM_RS"

# # ./build/$BIN_TEST > run-logs/plots/bench-xorshift-rs.txt

# # #Rejection Sampling + Tempering
# # make clean all BASE_CFLAGS+=" -DRANDOM_RS -DRANDOM_TEMPERING"

# # ./build/$BIN_TEST > run-logs/plots/bench-xorshift-rs-temp.txt

# #Low Bits (No Rejection Sampling) + Tempering
# make clean all BASE_CFLAGS+=" -DRANDOM_LOW_BITS -DRANDOM_TEMPERING"

# ./build/$BIN_TEST > run-logs/plots/bench-xorshift-low-bits.txt

# #Low Bits (No Rejection Sampling) + no-mul Tempering
# make clean all BASE_CFLAGS+=" -DRANDOM_LOW_BITS -DRANDOM_TEMPERING_NO_MUL"

# ./build/$BIN_TEST > run-logs/plots/bench-xorshift-low-bits-no-mul.txt

# #Modulo (No Rejection Sampling; No Tempering)
# make clean all

# ./build/$BIN_TEST > run-logs/plots/bench-xorshift-modulo.txt

# #Modulo + Tempering
# make clean all BASE_CFLAGS+=" -DRANDOM_TEMPERING"

# ./build/$BIN_TEST > run-logs/plots/bench-xorshift-modulo-temp.txt

# #Modulo + no-mul Tempering
# make clean all BASE_CFLAGS+=" -DRANDOM_TEMPERING_NO_MUL"

# ./build/$BIN_TEST > run-logs/plots/bench-xorshift-modulo-temp-no-mul.txt

# #Glibc
# make clean all BASE_CFLAGS+=" -DRANDOM_GLIBC"

# ./build/$BIN_TEST > run-logs/plots/bench-xorshift-glibc.txt







BIN_TESTS=(
    "nes/iris-nes"
    "ext/digits"
)

DEFS=(
  ""                                        # pure modulo, no temper, no low-bits
  "-DRANDOM_TEMPERING"                      # modulo + temper
  "-DRANDOM_TEMPERING_NO_MUL"               # modulo + temper-no-mul
  "-DRANDOM_LOW_BITS -DRANDOM_TEMPERING"    # low-bits + temper
  "-DRANDOM_LOW_BITS -DRANDOM_TEMPERING_NO_MUL" # low-bits + temper-no-mul
  "-DRANDOM_GLIBC"                          # glibc
)
LABELS=(
  "modulo"
  "modulo-temp"
  "modulo-temp-no-mul"
  "low-bits"
  "low-bits-no-mul"
  "glibc"
)

# Папка для логов
LOGDIR="run-logs/plots"

for j in "${!BIN_TESTS[@]}"; do
    BIN_TEST="${BIN_TESTS[j]}"
    mkdir -p "$LOGDIR"/"$BIN_TEST"
    echo "=== Building & running: $bin_test ==="
    for i in "${!DEFS[@]}"; do
        defs="${DEFS[i]}"
        label="${LABELS[i]}"

        echo "=== Building & running: $label ==="
        make clean all BASE_CFLAGS+=" $defs"
        ./build/"$BIN_TEST" > "$LOGDIR/$BIN_TEST/bench-xorshift-${label}.txt"
    done
done