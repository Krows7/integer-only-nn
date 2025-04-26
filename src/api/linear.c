#include "linear.h"
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ------------ Matrix8 ------------

#if ALLOW_LINEAR_VERBOSE > 0
// #undef stub
// #define get_mul8_2t stub
// #define init_m32_ stub

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

// ------------ Pool32 ------------

Pool32* create_pool_32(int capacity) {
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

// ------------ Pool8 ------------

Pool8* create_pool_8(lsize_t capacity) {
    Pool8* pool = malloc(sizeof(Pool8));
    pool->matrices = malloc(capacity * sizeof(Matrix8*));
    for (lsize_t i = 0; i < capacity; i++) {
        pool->matrices[i] = malloc(sizeof(Matrix8));
    }
    pool->reserved = calloc(sizeof(uint8_t), capacity);
    pool->size = 0;
    pool->capacity = capacity;
    return pool;
}

#define M8_CAPACITY 20

Pool8* m8_pool = NULL;

void init_pools() {
    m_pool = create_pool_32(M32_CAPACITY);
    m8_pool = create_pool_8(M8_CAPACITY);
}

// Matrix32* request_m32(lsize_t width, lsize_t height) {
//     if (!m_pool) m_pool = create_pool_32(M32_CAPACITY);
//     if (m_pool->size < m_pool->capacity) {
//         m_pool->reserved[m_pool->size] = 1;
//         *m_pool->matrices[m_pool->size] = init_m32(width, height);
//         return m_pool->matrices[m_pool->size++];
//     } else {
//         for (lsize_t i = 0; i < m_pool->size; i++) {
//             if (m_pool->reserved[i] == 0 && m_pool->matrices[i]->width == width && m_pool->matrices[i]->height == height) {
//                 m_pool->reserved[i] = 1;
//                 return m_pool->matrices[i];
//             }
//         }
//     }
//     // return init_m32(width, height);
// }

// void release_m32(Matrix32 *m) {

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

    free(m8_pool->reserved);
    for (lsize_t i = 0; i < m8_pool->size; i++) {
        for (lsize_t j = 0; j < m8_pool->matrices[i]->width; ++j) {
            free(m8_pool->matrices[i]->matrix[j]);
        }
        free(m8_pool->matrices[i]->matrix);
        free(m8_pool->matrices[i]);
    }
    free(m8_pool->matrices);
    free(m8_pool);
}

Matrix8 init_m8_(lsize_t width, lsize_t height) {
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

Matrix8 init_m8(lsize_t width, lsize_t height) {
    count_inc(d_request_m8);
    if (m8_pool->size < m8_pool->capacity) {
        m8_pool->reserved[m8_pool->size] = 1;
        *m8_pool->matrices[m8_pool->size] = init_m8_(width, height);
        return *m8_pool->matrices[m8_pool->size++];
    } else {
        uint32_t first_free = m8_pool->capacity;
        uint32_t first_suitable_free = m8_pool->capacity;
        for (lsize_t i = 0; i < m8_pool->size; i++) {
            if (m8_pool->reserved[i] == 0) {
                if (first_free == m8_pool->capacity) first_free = i;
                if (first_suitable_free == m8_pool->capacity && m8_pool->matrices[i]->height == height) first_suitable_free = i;
                if (m8_pool->matrices[i]->width == width && m8_pool->matrices[i]->height == height) {
                    m8_pool->reserved[i] = 1;
                    return *m8_pool->matrices[i];
                }
            }
        }
        if (first_suitable_free != m8_pool->capacity) {
            m8_pool->reserved[first_suitable_free] = 1;
            Matrix8* m = m8_pool->matrices[first_suitable_free];
            int8_t** matrix = malloc(width * sizeof(int8_t*));
            if (m->width < width) {
                for (lsize_t i = 0; i < m->width; ++i) {
                    matrix[i] = m->matrix[i];
                }
                for (lsize_t i = m->width; i < width; ++i) {
                    matrix[i] = malloc(height * sizeof(int8_t));
                }
            } else {
                for (lsize_t i = 0; i < width; ++i) {
                    matrix[i] = m->matrix[i];
                }
                for (lsize_t i = width; i < m->width; ++i) {
                    free(m->matrix[i]);
                }
            }
            free(m->matrix);
            m->matrix = matrix;
            m->width = width;
            return *m;
        }
        if (first_free != m8_pool->capacity) {
            m8_pool->reserved[first_free] = 1;
            for (lsize_t j = 0; j < m8_pool->matrices[first_free]->width; ++j) {
                free(m8_pool->matrices[first_free]->matrix[j]);
            }
            free(m8_pool->matrices[first_free]->matrix);
            *m8_pool->matrices[first_free] = init_m8_(width, height);
            return *m8_pool->matrices[first_free];
        }
    }
    return init_m8_(width, height);
}

void free_m8(Matrix8* m) {
    if (!m || !m->matrix) return;
    for (lsize_t i = 0; i < m8_pool->size; i++) {
        if (m8_pool->matrices[i]->matrix == m->matrix) {
            m8_pool->reserved[i] = 0;
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
    count_inc(d_m8_free);
}

void print_matrix8(const Matrix8* m, char* name) {
    print("[%s] Matrix:\n[", name);
    for (lsize_t i = 0; i < m->width; ++i) {
        print("[");
        for (lsize_t j = 0; j < m->height; ++j) {
            if (j == m->height - 1) print("%d", m->matrix[i][j]);
            else print("%d, ", m->matrix[i][j]);
        }
        if (i == m->width - 1) print("]");
        else println("]");
    }
    println("]");
    println("Scale: %d; Shape: (%d, %d)\n", m->scale, m->width, m->height);
    count_inc(d_m8_print);
}

Matrix32 init_m32_(lsize_t width, lsize_t height) {
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

// #if ALLOW_LINEAR_VERBOSE > 0
// #define stub(...) stub_1(__VA_ARGS__)
// #endif

// ------------ Matrix32 ------------
Matrix32 init_m32(lsize_t width, lsize_t height) {
    count_inc(d_request_m32);
    if (m_pool->size < m_pool->capacity) {
        m_pool->reserved[m_pool->size] = 1;
        *m_pool->matrices[m_pool->size] = init_m32_(width, height);
        return *m_pool->matrices[m_pool->size++];
    } else {
        uint32_t first_free = m_pool->capacity;
        uint32_t first_suitable_free = m_pool->capacity;
        for (lsize_t i = 0; i < m_pool->size; i++) {
            if (m_pool->reserved[i] == 0) {
                if (first_free == m_pool->capacity) first_free = i;
                if (first_suitable_free == m_pool->capacity && m_pool->matrices[i]->height == height) first_suitable_free = i;
                if (m_pool->matrices[i]->width == width && m_pool->matrices[i]->height == height) {
                    m_pool->reserved[i] = 1;
                    return *m_pool->matrices[i];
                }
            }
        }
        if (first_suitable_free != m_pool->capacity) {
            m_pool->reserved[first_suitable_free] = 1;
            Matrix32* m = m_pool->matrices[first_suitable_free];
            int32_t** matrix = malloc(width * sizeof(int32_t*));
            if (m->width < width) {
                for (lsize_t i = 0; i < m->width; ++i) {
                    matrix[i] = m->matrix[i];
                }
                for (lsize_t i = m->width; i < width; ++i) {
                    matrix[i] = malloc(height * sizeof(int32_t));
                }
            } else {
                for (lsize_t i = 0; i < width; ++i) {
                    matrix[i] = m->matrix[i];
                }
                for (lsize_t i = width; i < m->width; ++i) {
                    free(m->matrix[i]);
                }
            }
            free(m->matrix);
            m->matrix = matrix;
            m->width = width;
            return *m;
        }
        if (first_free != m_pool->capacity) {
            m_pool->reserved[first_free] = 1;
            for (lsize_t j = 0; j < m_pool->matrices[first_free]->width; ++j) {
                free(m_pool->matrices[first_free]->matrix[j]);
            }
            free(m_pool->matrices[first_free]->matrix);
            *m_pool->matrices[first_free] = init_m32_(width, height);
            return *m_pool->matrices[first_free];
        }
    }
    return init_m32_(width, height);
    // int32_t** matrix = malloc(width * sizeof(int32_t*));
    // for (lsize_t i = 0; i < width; ++i) {
    //     matrix[i] = malloc(height * sizeof(int32_t));
    // }
    // Matrix32 result;
    // result.width = width;
    // result.height = height;
    // result.matrix = matrix;
    // count_inc(d_m32_init);
    // return result;
}

void free_m32(Matrix32* m) {
    if (!m || !m->matrix) return;
    for (lsize_t i = 0; i < m_pool->size; i++) {
        if (m_pool->matrices[i]->matrix == m->matrix) {
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
    print("[%s] Matrix:\n[", name);
    for (lsize_t i = 0; i < m->width; ++i) {
        print("[");
        for (lsize_t j = 0; j < m->height; ++j) {
            if (j == m->height - 1) print("%d", m->matrix[i][j]);
            else print("%d, ", m->matrix[i][j]);
        }
        if (i == m->width - 1) print("]");
        else println("]");
    }
    println("]");
    println("Scale: %d", m->scale);
    println("Shape: (%d, %d)\n", m->width, m->height);
    count_inc(d_m32_print);
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
    print("[%s] Vector: [", name);
    for (uint8_t i = 0; i < v->length; ++i) {
        if (i == v->length - 1) print("%d", v->vector[i]);
        else print("%d, ", v->vector[i]);
    }
    println("]");
    println("Scale: %d", v->scale);
    println("Length: %d\n", v->length);
    count_inc(d_v8_print);
}

// ------------ Matrix Operations ------------

void mul8(const Matrix8* A, const Matrix8* B, Matrix32* C) {
    if (A->height != B->width) {
        println("Error: A->height != B->width: (%d != %d)", A->height, B->width);
        println("A: (%d, %d)", A->width, A->height);
        println("B: (%d, %d)", B->width, B->height);
        return;
    }
    if (A->width != C->width || B->height != C->height) {
        println("Error: A->width != C->width || B->height != C->height: (%d != %d) || (%d != %d)", A->width, C->width, B->height, C->height);
        println("A: (%d, %d)", A->width, A->height);
        println("B: (%d, %d)", B->width, B->height);
        println("C: (%d, %d)", C->width, C->height);
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
        println("Error in sub8: Matrix dimensions do not match.");
        println("M shape: (%d, %d), sub shape: (%d, %d)", M->width, M->height, sub->width, sub->height);
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
        println("Error: A_T->width != B->width: (%d != %d)", A_T->width, B->width);
        println("A_T: (%d, %d)", A_T->width, A_T->height);
        println("B: (%d, %d)", B->width, B->height);
        return;
    }
    if (A_T->height != C->width || B->height != C->height) {
        println("Error: A_T->height != C->width || B->height != C->height: (%d != %d) || (%d != %d)", A_T->height, C->width, B->height, C->height);
        println("A_T: (%d, %d)", A_T->width, A_T->height);
        println("B: (%d, %d)", B->width, B->height);
        println("C: (%d, %d)", C->width, C->height);
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
        println("Error: A->height != B_T->height: (%d != %d)", A->height, B_T->height);
        println("A: (%d, %d)", A->width, A->height);
        println("B_T: (%d, %d)", B_T->width, B_T->height);
        return;
    }
    if (A->width != C->width || B_T->width != C->height) {
        println("Error: A->width != C->width || B_T->width != C->height: (%d != %d) || (%d != %d)", A->width, C->width, B_T->width, C->height);
        println("A: (%d, %d)", A->width, A->height);
        println("B_T: (%d, %d)", B_T->width, B_T->height);
        println("C: (%d, %d)", C->width, C->height);
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

// #if ALLOW_LINEAR_VERBOSE > 0
// #define stub(...) stub_1(__VA_ARGS__)
// #endif

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