cd /home/huy/network-project/

# Run from /home/huy/network-project/
# vc -o test.nes \
#    src/tests/nes/iris-nes.c \
#    src/api/linear.c \
#    src/api/network.c \
#    src/api/quantization.c \
#    src/api/base.c \
#    src/nes/dataset_bench_int.c \
#    -Isrc/nes -Isrc/api -DNES -DVBCC -O4 -size -final +nromnew

vc +tkrom -o test.nes \
   src/tests/nes/iris-nes.c \
   src/api/linear.c \
   src/api/network.c \
   src/api/quantization.c \
   src/api/base.c \
   src/nes/dataset_bench_int.c \
   -Isrc/nes -Isrc/api -DNES -DVBCC -size -final -O4 -stack-check -force-statics

# Flags +unrom512v -O=65535