#ifndef LINEAR_64_H
#define LINEAR_64_H

#include "linear.h"

typedef struct {
    int64_t** matrix;
    lsize_t width;
    lsize_t height;
    int8_t scale;
} Matrix64;

// Matrix64
Matrix64 init_m64(lsize_t width, lsize_t height);
void free_m64(Matrix64* m);
void print_matrix64(const Matrix64* m, char* name);

#endif