#include "linear.h"
#include <inttypes.h>

// ------------ Matrix8 ------------

Matrix8 init_m8(uint8_t width, uint8_t height) {
    int8_t** matrix = malloc(width * sizeof(int8_t*));
    for (uint8_t i = 0; i < width; ++i) {
        matrix[i] = malloc(height * sizeof(int8_t));
        // for (uint8_t j = 0; j < height; ++j) {
        //     matrix[i][j] = 0;
        // }
    }
    Matrix8 result;
    result.matrix = matrix;
    result.width = width;
    result.height = height;
    result.scale = 0;
    return result;
}

void free_m8(Matrix8* m) {
    if (!m || !m->matrix) return;
    for (uint8_t i = 0; i < m->width; ++i) {
        free(m->matrix[i]);
    }
    free(m->matrix);
    m->matrix = NULL;
    m->width = 0;
    m->height = 0;
    m->scale = 0;
}

void print_matrix8(Matrix8* m, char* name) {
    printf("[%s] Matrix:\n[", name);
    for (uint8_t i = 0; i < m->width; ++i) {
        printf("[");
        for (uint8_t j = 0; j < m->height; ++j) {
            if (j == m->height - 1) printf("%d", m->matrix[i][j]);
            else printf("%d, ", m->matrix[i][j]);
        }
        if (i == m->width - 1) printf("]");
        else printf("]\n");
    }
    printf("]\n");
    printf("Scale: %d; Shape: (%d, %d)\n\n", m->scale, m->width, m->height);
}

// ------------ Matrix32 ------------

Matrix32 init_m32(uint8_t width, uint8_t height) {
    int32_t** matrix = malloc(width * sizeof(int32_t*));
    for (uint8_t i = 0; i < width; ++i) {
        matrix[i] = malloc(height * sizeof(int32_t));
        // for (uint8_t j = 0; j < height; ++j) {
        //     matrix[i][j] = 0;
        // }
    }
    Matrix32 result;
    result.width = width;
    result.height = height;
    result.matrix = matrix;
    result.scale = 0;
    return result;
}

void free_m32(Matrix32* m) {
    if (!m || !m->matrix) return;
    for (uint8_t i = 0; i < m->width; ++i) {
        free(m->matrix[i]);
    }
    free(m->matrix);
    m->matrix = NULL;
    m->width = 0;
    m->height = 0;
    m->scale = 0;
}

void print_matrix32(Matrix32* m, char* name) {
    printf("[%s] Matrix:\n[", name);
    for (uint8_t i = 0; i < m->width; ++i) {
        printf("[");
        for (uint8_t j = 0; j < m->height; ++j) {
            if (j == m->height - 1) printf("%d", m->matrix[i][j]);
            else printf("%d, ", m->matrix[i][j]);
        }
        if (i == m->width - 1) printf("]");
        else printf("]\n");
    }
    printf("]\n");
    printf("Scale: %d\n", m->scale);
    printf("Shape: (%d, %d)\n\n", m->width, m->height);
}

// ------------ Matrix64 ------------

void print_matrix64(Matrix64* m, char* name) {
    printf("[%s] Matrix:\n[", name);
    for (uint8_t i = 0; i < m->width; ++i) {
        printf("[");
        for (uint8_t j = 0; j < m->height; ++j) {
            if (j == m->height - 1) printf("%" PRId64, m->matrix[i][j]);
            else printf("%" PRId64 ", ", m->matrix[i][j]);
        }
        if (i == m->width - 1) printf("]");
        else printf("]\n");
    }
    printf("]\n");
    printf("Scale: %d\n", m->scale);
    printf("Shape: (%d, %d)\n\n", m->width, m->height);
}

Matrix64 init_m64(uint8_t width, uint8_t height) {
    int64_t** matrix = malloc(width * sizeof(int64_t*));
    for (uint8_t i = 0; i < width; ++i) {
        matrix[i] = malloc(height * sizeof(int64_t));
        // for (uint8_t j = 0; j < height; ++j) {
        //     matrix[i][j] = 0;
        // }
    }
    Matrix64 result;
    result.matrix = matrix;
    result.width = width;
    result.height = height;
    result.scale = 0;
    return result;
}

void free_m64(Matrix64* m) {
    if (!m || !m->matrix) return;
    for (uint8_t i = 0; i < m->width; ++i) {
        free(m->matrix[i]);
    }
    free(m->matrix);
    m->matrix = NULL;
    m->width = 0;
    m->height = 0;
    m->scale = 0;
}

// ------------ Vector8 ------------

Vector8 init_v8(uint8_t length) {
    int8_t* vector = malloc(length * sizeof(int8_t));
    if (vector == NULL) return (Vector8){NULL, 0, 0};
    // for (uint8_t i = 0; i < length; ++i) {
    //     vector[i] = 0;
    // }
    Vector8 result;
    result.vector = vector;
    result.length = length;
    result.scale = 0;
    return result;
}

void free_v8(Vector8* v) {
    if (!v || !v->vector) return;
    free(v->vector);
    v->vector = NULL;
    v->length = 0;
    v->scale = 0;
}

void print_vector8(Vector8* v, char* name) {
    printf("[%s] Vector: [", name);
    for (uint8_t i = 0; i < v->length; ++i) {
        if (i == v->length - 1) printf("%d", v->vector[i]);
        else printf("%d, ", v->vector[i]);
    }
    printf("]\n");
    printf("Scale: %d\n", v->scale);
    printf("Length: %d\n\n", v->length);
}

// Matrix8_ext

Matrix8_ext init_m8_ext(uint32_t width, uint32_t height) {
    int8_t** matrix = malloc(width * sizeof(int8_t*));
    if (matrix == NULL) return (Matrix8_ext){NULL, 0, 0, 0};
    for (uint32_t i = 0; i < width; ++i) {
        matrix[i] = malloc(height * sizeof(int8_t));
        // for (uint32_t j = 0; j < height; ++j) {
        //     matrix[i][j] = 0;
        // }
    }
    Matrix8_ext result;
    result.matrix = matrix;
    result.width = width;
    result.height = height;
    result.scale = 0;
    return result;
}

void free_m8_ext(Matrix8_ext* m) {
    if (!m || !m->matrix) return;
    for (uint32_t i = 0; i < m->width; ++i) {
        free(m->matrix[i]);
    }
    free(m->matrix);
    m->matrix = NULL;
    m->width = 0;
    m->height = 0;
    m->scale = 0;
}

void print_matrix8_ext(Matrix8_ext* m, char* name) {
    printf("[%s] Matrix:\n[", name);
    for (uint32_t i = 0; i < m->width; ++i) {
        printf("[");
        for (uint32_t j = 0; j < m->height; ++j) {
            if (j == m->height - 1) printf("%d", m->matrix[i][j]);
            else printf("%d, ", m->matrix[i][j]);
        }
        if (i == m->width - 1) printf("]");
        else printf("]\n");
    }
    printf("]\n");
    printf("Scale: %d; Shape: (%d, %d)\n\n", m->scale, m->width, m->height);
}

// Vector8_ext

Vector8_ext init_v8_ext(uint32_t length) {
    int8_t* vector = malloc(length * sizeof(int8_t));
    if (vector == NULL) return (Vector8_ext){NULL, 0, 0};
    for (uint32_t i = 0; i < length; ++i) {
        vector[i] = 0;
    }
    Vector8_ext result;
    result.vector = vector;
    result.length = length;
    result.scale = 0;
    return result;
}

void free_v8_ext(Vector8_ext* v) {
    if (!v || !v->vector) return;
    free(v->vector);
    v->vector = NULL;
    v->length = 0;
    v->scale = 0;
}

void print_vector8_ext(Vector8_ext* v, char* name) {
    printf("[%s] Vector: [", name);
    for (uint32_t i = 0; i < v->length; ++i) {
        if (i == v->length - 1) printf("%d", v->vector[i]);
        else printf("%d, ", v->vector[i]);
    }
    printf("]\n");
    printf("Scale: %d\n", v->scale);
    printf("Length: %d\n\n", v->length);
}


// ------------ Matrix Operations ------------

void mul8(const Matrix8* A, const Matrix8* B, Matrix32* C) {
    if (A->height != B->width) {
        printf("Error: A->height != B->width: (%d != %d)\n", A->height, B->width);
        printf("A: (%d, %d)\n", A->width, A->height);
        printf("B: (%d, %d)\n", B->width, B->height);
        return;
    }
    if (A->width != C->width || B->height != C->height) {
        printf("Error: A->width != C->width || B->height != C->height: (%d != %d) || (%d != %d)\n", A->width, C->width, B->height, C->height);
        printf("A: (%d, %d)\n", A->width, A->height);
        printf("B: (%d, %d)\n", B->width, B->height);
        printf("C: (%d, %d)\n", C->width, C->height);
        return;
    }

    for (uint8_t i = 0; i < A->width; ++i) {
        for (uint8_t j = 0; j < B->height; ++j) {
            C->matrix[i][j] = 0;
            for (uint8_t k = 0; k < A->height; k++) {
                C->matrix[i][j] += A->matrix[i][k] * B->matrix[k][j];
            }
        }
    }

    C->scale = A->scale + B->scale;
}

Matrix32 get_mul8(const Matrix8* A, const Matrix8* B) {
    Matrix32 C = init_m32(A->width, B->height);
    mul8(A, B, &C);
    return C;
}

Matrix32 get_mul32(const Matrix32* A, const Matrix8* B) {
    Matrix32 C = init_m32(A->width, B->height);
    for (uint8_t i = 0; i < A->width; ++i) {
        for (uint8_t j = 0; j < B->height; ++j) {
            C.matrix[i][j] = 0;
            for (uint8_t k = 0; k < A->height; k++) {
                C.matrix[i][j] += A->matrix[i][k] * B->matrix[k][j];
            }
        }
    }
    C.scale = A->scale + B->scale;
    return C;
}

Matrix32 to_mat32(Matrix8* m) {
    Matrix32 result = init_m32(m->width, m->height);
    for (uint8_t i = 0; i < m->width; ++i) {
        for (uint8_t j = 0; j < m->height; ++j) {
            result.matrix[i][j] = m->matrix[i][j] << m->scale;
        }
    }
    return result;
}

void sub8(Matrix8* M, const Matrix8* sub) {
    if (M->width != sub->width || M->height != sub->height) {
        fprintf(stderr, "Error in sub8: Matrix dimensions do not match.\n");
        fprintf(stderr, "M shape: (%d, %d), sub shape: (%d, %d)\n",
                M->width, M->height, sub->width, sub->height);
        exit(1);
    }

    for (uint8_t i = 0; i < M->width; ++i) {
        for (uint8_t j = 0; j < M->height; ++j) {
            M->matrix[i][j] -= sub->matrix[i][j];
        }
    }
}

// A must be transposed in product
void mul8_1t(const Matrix8* A_T, const Matrix8* B, Matrix32* C) {
    if (A_T->width != B->width) {
        printf("Error: A_T->width != B->width: (%d != %d)\n", A_T->width, B->width);
        printf("A_T: (%d, %d)\n", A_T->width, A_T->height);
        printf("B: (%d, %d)\n", B->width, B->height);
        return;
    }
    if (A_T->height != C->width || B->height != C->height) {
        printf("Error: A_T->height != C->width || B->height != C->height: (%d != %d) || (%d != %d)\n", A_T->height, C->width, B->height, C->height);
        printf("A_T: (%d, %d)\n", A_T->width, A_T->height);
        printf("B: (%d, %d)\n", B->width, B->height);
        printf("C: (%d, %d)\n", C->width, C->height);
        return;
    }
    
    for (uint8_t i = 0; i < A_T->height; ++i) {
        for (uint8_t j = 0; j < B->height; ++j) {
            C->matrix[i][j] = 0;
            for (uint8_t k = 0; k < A_T->width; k++) {
                C->matrix[i][j] += A_T->matrix[k][i] * B->matrix[k][j];
            }
        }
    }

    C->scale = A_T->scale + B->scale;
}

Matrix32 get_mul8_1t(const Matrix8* A_T, const Matrix8* B) {
    Matrix32 C = init_m32(A_T->height, B->height);
    mul8_1t(A_T, B, &C);
    return C;
}

// B must be transposed in product
void mul8_2t(const Matrix8* A, const Matrix8* B_T, Matrix32* C) {
    if (A->height != B_T->height) {
        printf("Error: A->height != B_T->height: (%d != %d)\n", A->height, B_T->height);
        printf("A: (%d, %d)\n", A->width, A->height);
        printf("B_T: (%d, %d)\n", B_T->width, B_T->height);
        return;
    }
    if (A->width != C->width || B_T->width != C->height) {
        printf("Error: A->width != C->width || B_T->width != C->height: (%d != %d) || (%d != %d)\n", A->width, C->width, B_T->width, C->height);
        printf("A: (%d, %d)\n", A->width, A->height);
        printf("B_T: (%d, %d)\n", B_T->width, B_T->height);
        printf("C: (%d, %d)\n", C->width, C->height);
        return;
    }

    for (uint8_t i = 0; i < A->width; ++i) {
        for (uint8_t j = 0; j < B_T->width; ++j) {
            C->matrix[i][j] = 0;
            for (uint8_t k = 0; k < A->height; k++) {
                C->matrix[i][j] += A->matrix[i][k] * B_T->matrix[j][k];
            }
        }
    }
    C->scale = A->scale + B_T->scale;
}

Matrix32 get_mul8_2t(const Matrix8* A, const Matrix8* B_T) {
    Matrix32 C = init_m32(A->width, B_T->width);
    mul8_2t(A, B_T, &C);
    return C;
}

// ------------ Other ------------

uint8_t ceil_log_2(int32_t value) {
    uint8_t result = 1;
    while (value > 1) {
        value >>= 1;
        ++result;
    }
    return result;
}

void relu8(Matrix8* m) {
    for (uint8_t i = 0; i < m->width; ++i) {
        for (uint8_t j = 0; j < m->height; ++j) {
            if (m->matrix[i][j] < 0) {
                m->matrix[i][j] = 0;
            }
        }
    }
}

uint8_t effective_bitwidth(Matrix32* matrix) {
    int32_t max_value = 0;
    for (uint8_t i = 0; i < matrix->width; ++i) {
        for (uint8_t j = 0; j < matrix->height; ++j) {
            if (abs(matrix->matrix[i][j]) > max_value) max_value = abs(matrix->matrix[i][j]);
            // if (matrix->matrix[i][j] > max_value) max_value = matrix->matrix[i][j];
            // else if (matrix->matrix[i][j] < 0 && -matrix->matrix[i][j] > max_value) max_value = -matrix->matrix[i][j];
        }
    }
    uint8_t b = ceil_log_2(max_value);
    return b;
}