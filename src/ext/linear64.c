#include "linear64.h"
#include <inttypes.h>
#include <stdio.h>

// ------------ Matrix64 ------------

void print_matrix64(const Matrix64* m, char* name) {
    print("[%s] Matrix:\n[", name);
    for (lsize_t i = 0; i < m->width; ++i) {
        print("[");
        for (lsize_t j = 0; j < m->height; ++j) {
            if (j == m->height - 1) print("%" PRId64, m->matrix[i][j]);
            else print("%" PRId64 ", ", m->matrix[i][j]);
        }
        if (i == m->width - 1) print("]");
        else println("]");
    }
    println("]");
    println("Scale: %d", m->scale);
    println("Shape: (%d, %d)\n", m->width, m->height);
    count_inc(d_m64_print);
}

Matrix64 init_m64(lsize_t width, lsize_t height) {
    int64_t** matrix = malloc(width * sizeof(int64_t*));
    for (lsize_t i = 0; i < width; ++i) {
        matrix[i] = malloc(height * sizeof(int64_t));
    }
    Matrix64 result;
    result.matrix = matrix;
    result.width = width;
    result.height = height;
    count_inc(d_m64_init);
    return result;
}

void free_m64(Matrix64* m) {
    if (!m || !m->matrix) return;
    for (lsize_t i = 0; i < m->width; ++i) {
        free(m->matrix[i]);
    }
    free(m->matrix);
    m->matrix = NULL;
    m->width = 0;
    m->height = 0;
    m->scale = 0;
    count_inc(d_m64_free);
}