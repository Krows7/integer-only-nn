#include "linear.h"
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ------------ Matrix8 ------------

#if ALLOW_LINEAR_VERBOSE > 0
#undef stub
#define get_mul8_2t stub
lin_count(d_m8_init);
lin_count(d_m8_free);
lin_count(d_m8_print);
lin_count(d_m32_init);
lin_count(d_m32_free);
lin_count(d_m32_print);
lin_count(d_m64_init);
lin_count(d_m64_free);
lin_count(d_m64_print);
lin_count(d_v8_init);
lin_count(d_v8_free);
lin_count(d_v8_print);
lin_count(d_mul8);
lin_count(d_get_mul8);
lin_count(d_get_mul32);
lin_count(d_to_mat32);
lin_count(d_sub8);
lin_count(d_mul8_1t);
lin_count(d_get_mul8_1t);
lin_count(d_mul8_2t);
lin_count(d_get_mul8_2t);
lin_count(d_ceil_log_2);
lin_count(d_relu8);
lin_count(d_effective_bitwidth);
lin_count(d_print_metrics);

#define MAX_SIZE 100 // Maximum number of elements in the map 
  
int size = 0; // Current number of elements in the map 
char keys[MAX_SIZE][100]; // Array to store the keys 
int values[MAX_SIZE]; // Array to store the values 
  
// Function to get the index of a key in the keys array 
int getIndex(const char key[]) 
{ 
    for (int i = 0; i < size; i++) { 
        if (strcmp(keys[i], key) == 0) { 
            return i; 
        } 
    } 
    return -1; // Key not found 
} 
  
// Function to insert a key-value pair into the map 
void insert(const char key[], int value) 
{ 
    int index = getIndex(key); 
    if (index == -1) { // Key not found 
        strcpy(keys[size], key); 
        values[size] = value; 
        size++; 
    } 
    else { // Key found 
        values[index] = value; 
    } 
} 
  
// Function to get the value of a key in the map 
int get(const char key[]) 
{ 
    int index = getIndex(key); 
    if (index == -1) { // Key not found 
        return 0; 
    } 
    else { // Key found 
        return values[index]; 
    } 
} 
  
// Function to print the map 
void printMap() 
{ 
    for (int i = 0; i < size; i++) { 
        printf("%s: %d\n", keys[i], values[i]); 
    } 
}
#endif

Pool32* create_pool(int capacity) {
    Pool32* pool = malloc(sizeof(Pool32));
    pool->matrices = malloc(capacity * sizeof(Matrix32*));
    for (int i = 0; i < capacity; i++) {
        pool->matrices[i] = malloc(sizeof(Matrix32));
    }
    pool->reserved = calloc(sizeof(uint8_t), capacity);
    pool->size = 0;
    pool->capacity = capacity;
    return pool;
}

#define M32_CAPACITY 20

Pool32* m_pool = NULL;

void init_pools() {
    m_pool = create_pool(M32_CAPACITY);
}

Matrix32* request_m32(lsize_t width, lsize_t height) {
    if (!m_pool) m_pool = create_pool(M32_CAPACITY);
    if (m_pool->size < m_pool->capacity) {
        m_pool->reserved[m_pool->size] = 1;
        *m_pool->matrices[m_pool->size] = init_m32(width, height);
        return m_pool->matrices[m_pool->size++];
    } else {
        for (lsize_t i = 0; i < m_pool->size; i++) {
            // log("%d %d %d %d %d", m_pool->reserved[i], m_pool->matrices[i]->width, width, m_pool->matrices[i]->height, height);
            if (m_pool->reserved[i] == 0 && m_pool->matrices[i]->width == width && m_pool->matrices[i]->height == height) {
                m_pool->reserved[i] = 1;
                return m_pool->matrices[i];
            }
        }
    }
    // return init_m32(width, height);
}

void release_m32(Matrix32 *m) {

}

// void free_pool(Pool32* pool) {
//     for (int i = 0; i < pool->size; i++) {
//         free_m32(pool->matrices[i]);
//     }
//     free(pool->matrices);
//     free(pool);
// }

// void fit_pool(Pool32* pool) {
//     for (int i = 1; i < pool->size; i++) {
//         if (pool->matrices[i - 1] == NULL && pool->matrices[i] != NULL) {
//             pool->matrices[i - 1] = pool->matrices[i];
//             pool->matrices[i] = NULL;
//         }
//     }
// }

// Matrix32* get_matrix(Pool32* pool, lsize_t width, lsize_t height) {
//     if (pool->size < pool->capacity) {
//         Matrix32* m = malloc(sizeof(Matrix32));
//         *m = init_m32(width, height);
//         pool->matrices[pool->size++] = m;
//         return m;
//     } else {
//         return NULL;
//     }
// }

// // Matrix32* find_index(Pool32* pool, Matrix32* m) 

// void free_pool_matrix(Pool32* pool, Matrix32* matrix) {
//     for (int i = 0; i < pool->size; i++) {
//         if (pool->matrices[i] == matrix) {
//             free_m32(matrix);
//             pool->matrices[i] = NULL;
//             fit_pool(pool);
//             pool->size--;
//             return;
//         }
//     }
// }

void lin_cleanup() {
    free(m_pool->reserved);
    for (lsize_t i = 0; i < m_pool->size; i++) {
        for (lsize_t j = 0; j < m_pool->matrices[i]->width; ++j) {
            free(m_pool->matrices[i]->matrix[j]);
        }
        free(m_pool->matrices[i]->matrix);
        free(m_pool->matrices[i]);
    }
    free(m_pool->matrices);
    free(m_pool);
}

Matrix8 init_m8(lsize_t width, lsize_t height) {
    int8_t** matrix = malloc(width * sizeof(int8_t*));
    for (lsize_t i = 0; i < width; ++i) {
        matrix[i] = malloc(height * sizeof(int8_t));
    }
    Matrix8 result;
    result.matrix = matrix;
    result.width = width;
    result.height = height;
    count_inc(d_m8_init);
    return result;
}

void free_m8(Matrix8* m) {
    if (!m || !m->matrix) return;
    for (lsize_t i = 0; i < m->width; ++i) {
        free(m->matrix[i]);
    }
    free(m->matrix);
    m->matrix = NULL;
    m->width = 0;
    m->height = 0;
    m->scale = 0;
    count_inc(d_m8_free);
}

void print_matrix8(const Matrix8* m, char* name) {
    printf("[%s] Matrix:\n[", name);
    for (lsize_t i = 0; i < m->width; ++i) {
        printf("[");
        for (lsize_t j = 0; j < m->height; ++j) {
            if (j == m->height - 1) printf("%d", m->matrix[i][j]);
            else printf("%d, ", m->matrix[i][j]);
        }
        if (i == m->width - 1) printf("]");
        else printf("]\n");
    }
    printf("]\n");
    printf("Scale: %d; Shape: (%d, %d)\n\n", m->scale, m->width, m->height);
    count_inc(d_m8_print);
}

// ------------ Matrix32 ------------
Matrix32 init_m32(lsize_t width, lsize_t height) {
    int32_t** matrix = malloc(width * sizeof(int32_t*));
    for (lsize_t i = 0; i < width; ++i) {
        matrix[i] = malloc(height * sizeof(int32_t));
    }
    Matrix32 result;
    result.width = width;
    result.height = height;
    result.matrix = matrix;
    count_inc(d_m32_init);
    return result;
}

void free_m32(Matrix32* m) {
    if (!m || !m->matrix) return;
    for (int i = 0; i < m_pool->size; i++) {
        // log("%d %d", m_pool->matrices[i]->matrix, m->matrix);
        if (m_pool->matrices[i]->matrix == m->matrix) {
            // log("gre");
            m_pool->reserved[i] = 0;
            return;
        }
    }

    for (lsize_t i = 0; i < m->width; ++i) {
        free(m->matrix[i]);
    }
    free(m->matrix);
    m->matrix = NULL;
    m->width = 0;
    m->height = 0;
    m->scale = 0;
    count_inc(d_m32_free);
}

void print_matrix32(const Matrix32* m, char* name) {
    printf("[%s] Matrix:\n[", name);
    for (lsize_t i = 0; i < m->width; ++i) {
        printf("[");
        for (lsize_t j = 0; j < m->height; ++j) {
            if (j == m->height - 1) printf("%d", m->matrix[i][j]);
            else printf("%d, ", m->matrix[i][j]);
        }
        if (i == m->width - 1) printf("]");
        else printf("]\n");
    }
    printf("]\n");
    printf("Scale: %d\n", m->scale);
    printf("Shape: (%d, %d)\n\n", m->width, m->height);
    count_inc(d_m32_print);
}

// ------------ Matrix64 ------------

void print_matrix64(const Matrix64* m, char* name) {
    printf("[%s] Matrix:\n[", name);
    for (lsize_t i = 0; i < m->width; ++i) {
        printf("[");
        for (lsize_t j = 0; j < m->height; ++j) {
            if (j == m->height - 1) printf("%" PRId64, m->matrix[i][j]);
            else printf("%" PRId64 ", ", m->matrix[i][j]);
        }
        if (i == m->width - 1) printf("]");
        else printf("]\n");
    }
    printf("]\n");
    printf("Scale: %d\n", m->scale);
    printf("Shape: (%d, %d)\n\n", m->width, m->height);
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

// ------------ Vector8 ------------

Vector8 init_v8(lsize_t length) {
    int8_t* vector = malloc(length * sizeof(int8_t));
    if (vector == NULL) return (Vector8){NULL, 0, 0};
    Vector8 result;
    result.vector = vector;
    result.length = length;
    count_inc(d_v8_init);
    return result;
}

void free_v8(Vector8* v) {
    if (!v || !v->vector) return;
    free(v->vector);
    v->vector = NULL;
    v->length = 0;
    v->scale = 0;
    count_inc(d_v8_free);
}

void print_vector8(const Vector8* v, char* name) {
    printf("[%s] Vector: [", name);
    for (uint8_t i = 0; i < v->length; ++i) {
        if (i == v->length - 1) printf("%d", v->vector[i]);
        else printf("%d, ", v->vector[i]);
    }
    printf("]\n");
    printf("Scale: %d\n", v->scale);
    printf("Length: %d\n\n", v->length);
    count_inc(d_v8_print);
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

    for (lsize_t i = 0; i < A->width; ++i) {
        for (lsize_t j = 0; j < B->height; ++j) {
            C->matrix[i][j] = 0;
            for (lsize_t k = 0; k < A->height; k++) {
                C->matrix[i][j] += A->matrix[i][k] * B->matrix[k][j];
            }
        }
    }

    C->scale = A->scale + B->scale;
    count_inc(d_mul8);
}

Matrix32 get_mul8(const Matrix8* A, const Matrix8* B) {
    Matrix32 C = init_m32(A->width, B->height);
    mul8(A, B, &C);
    count_inc(d_get_mul8);
    return C;
}

Matrix32 get_mul32(const Matrix32* A, const Matrix8* B) {
    Matrix32 C = init_m32(A->width, B->height);
    for (lsize_t i = 0; i < A->width; ++i) {
        for (lsize_t j = 0; j < B->height; ++j) {
            C.matrix[i][j] = 0;
            for (lsize_t k = 0; k < A->height; k++) {
                C.matrix[i][j] += A->matrix[i][k] * B->matrix[k][j];
            }
        }
    }
    C.scale = A->scale + B->scale;
    count_inc(d_get_mul32);
    return C;
}

Matrix32 to_mat32(const Matrix8* m) {
    Matrix32 result = init_m32(m->width, m->height);
    for (lsize_t i = 0; i < m->width; ++i) {
        for (lsize_t j = 0; j < m->height; ++j) {
            result.matrix[i][j] = m->matrix[i][j] << m->scale;
        }
    }
    result.scale = m->scale;
    count_inc(d_to_mat32);
    return result;
}

void sub8(const Matrix8* M, const Matrix8* sub) {
    if (M->width != sub->width || M->height != sub->height) {
        fprintf(stderr, "Error in sub8: Matrix dimensions do not match.\n");
        fprintf(stderr, "M shape: (%d, %d), sub shape: (%d, %d)\n",
                M->width, M->height, sub->width, sub->height);
        exit(1);
    }

    for (lsize_t i = 0; i < M->width; ++i) {
        for (lsize_t j = 0; j < M->height; ++j) {
            M->matrix[i][j] -= sub->matrix[i][j];
        }
    }
    count_inc(d_sub8);
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
    
    for (lsize_t i = 0; i < A_T->height; ++i) {
        for (lsize_t j = 0; j < B->height; ++j) {
            C->matrix[i][j] = 0;
            for (lsize_t k = 0; k < A_T->width; k++) {
                C->matrix[i][j] += A_T->matrix[k][i] * B->matrix[k][j];
            }
        }
    }

    C->scale = A_T->scale + B->scale;
    count_inc(d_mul8_1t);
}

Matrix32 get_mul8_1t(const Matrix8* A_T, const Matrix8* B) {
    Matrix32 C = init_m32(A_T->height, B->height);
    mul8_1t(A_T, B, &C);
    count_inc(d_get_mul8_1t);
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

    for (lsize_t i = 0; i < A->width; ++i) {
        for (lsize_t j = 0; j < B_T->width; ++j) {
            C->matrix[i][j] = 0;
            for (lsize_t k = 0; k < A->height; k++) {
                C->matrix[i][j] += A->matrix[i][k] * B_T->matrix[j][k];
            }
        }
    }
    C->scale = A->scale + B_T->scale;
    count_inc(d_mul8_2t);
}

Matrix32 get_mul8_2t(const Matrix8* A, const Matrix8* B_T) {
    // log("%d %d", A->width, B_T->width);
    // if (!pool) pool = malloc(sizeof(Matrix32*));
    // Matrix32 C;
    // if (!*pool || !(*pool)->matrix) pool_reserved = 0;
    // if (!pool_reserved) {
    //     log("1");
    //     log("%s", pool);
    //     if (!pool) {
    //         log("2");
    //         // *pool = malloc(sizeof(Matrix32));
    //         *pool = init_m32(A->width, B_T->width);
    //         C = *pool;
    //     } else {
    //         if (pool->width != A->width || pool->height != B_T->width) {
    //             free_m32(pool);
    //             pool = malloc(sizeof(Matrix32));
    //             *pool = init_m32(A->width, B_T->width);
    //         }
    //         C = *pool;
    //     }
    //     pool_reserved = 1;
    // } else {
    //     C = init_m32(A->width, B_T->width);
    // }

    Matrix32 C = init_m32(A->width, B_T->width);
    mul8_2t(A, B_T, &C);
    count_inc(d_get_mul8_2t);
    return C;
}

#if ALLOW_LINEAR_VERBOSE > 0
#define stub(...) stub_1(__VA_ARGS__)
#endif

// ------------ Other ------------

uint8_t ceil_log_2(int32_t value) {
    uint8_t result = 1;
    while (value > 1) {
        value >>= 1;
        ++result;
    }
    count_inc(d_ceil_log_2);
    return result;
}

void relu8(const Matrix8* m) {
    for (lsize_t i = 0; i < m->width; ++i) {
        for (lsize_t j = 0; j < m->height; ++j) {
            if (m->matrix[i][j] < 0) {
                m->matrix[i][j] = 0;
            }
        }
    }
    count_inc(d_relu8);
}

uint8_t effective_bitwidth(const Matrix32* matrix) {
    int32_t max_value = 0;
    for (lsize_t i = 0; i < matrix->width; ++i) {
        for (lsize_t j = 0; j < matrix->height; ++j) {
            if (abs(matrix->matrix[i][j]) > max_value) max_value = abs(matrix->matrix[i][j]);
            // if (matrix->matrix[i][j] > max_value) max_value = matrix->matrix[i][j];
            // else if (matrix->matrix[i][j] < 0 && -matrix->matrix[i][j] > max_value) max_value = -matrix->matrix[i][j];
        }
    }
    uint8_t b = ceil_log_2(max_value);
    count_inc(d_effective_bitwidth);
    return b;
}

void print_metrics() {
    #if ALLOW_LINEAR_VERBOSE > 0
    log("Linear Math Metrics:");
    print_count(d_m8_init);
    print_count(d_m8_free);
    print_count(d_m8_print);
    print_count(d_m32_init);
    print_count(d_m32_free);
    print_count(d_m32_print);
    print_count(d_m64_init);
    print_count(d_m64_free);
    print_count(d_m64_print);
    print_count(d_v8_init);
    print_count(d_v8_free);
    print_count(d_v8_print);
    print_count(d_mul8);
    print_count(d_get_mul8);
    print_count(d_get_mul32);
    print_count(d_to_mat32);
    print_count(d_sub8);
    print_count(d_mul8_1t);
    print_count(d_get_mul8_1t);
    print_count(d_mul8_2t);
    print_count(d_get_mul8_2t);
    print_count(d_ceil_log_2);
    print_count(d_relu8);
    print_count(d_effective_bitwidth);
    print_count(d_print_metrics);
    #endif
}