#include "dataset_bench.h"
#include "float_ops.h"
#include "linear.h"
#include "stdio.h"

int main(void) {
    init_pools();
    float** m = (float*[1]) {NULL};
    // float** m = malloc(sizeof(float*));
    *m = (float[2]) {2.0f, 1.0f};
    // *m = malloc(2 * sizeof(float));
    // m[0][0] = 2;
    Matrix8 r = quantize_float_matrix_adaptive(m, 1, 2);
    print_matrix8(&r, "Rounded");
    // log("%d", -1 << 2);
    // log("%d", -64 >> 2);
    lin_cleanup();
}