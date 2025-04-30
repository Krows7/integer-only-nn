#ifndef NN_MATH_H
#define NN_MATH_H

#include "base.h"

// ------------ Struct Definitions ------------
typedef struct {
    int8_t** matrix;
    lsize_t width;
    lsize_t height;
    int8_t scale;
} Matrix8;

typedef struct {
    int32_t** matrix;
    lsize_t width;
    lsize_t height;
    int8_t scale;
} Matrix32;

typedef struct {
    int8_t* vector;
    lsize_t length;
    int8_t scale;
} Vector8;

typedef struct Pool32 {
    Matrix32** matrices;
    uint8_t* reserved;
    lsize_t size;
    lsize_t capacity;
} Pool32;

typedef struct Pool8 {
    Matrix8** matrices;
    uint8_t* reserved;
    lsize_t size;
    lsize_t capacity;
} Pool8;

__bank(2) void init_pools();

// If there's no room in Pool8, calls init_m8
__bank(2) Matrix8* request_m8(lsize_t width, lsize_t height);
__bank(2) void release_m8(Matrix8* m);

// If there's no room in Pool32, calls init_m32
__bank(2) Matrix32* request_m32(lsize_t width, lsize_t height);
__bank(2) void release_m32(Matrix32* m);

// ------------ Function Prototypes ------------

// Matrix8
__bank(2) Matrix8 init_m8(lsize_t width, lsize_t height);
__bank(2) void free_m8(Matrix8* m);
__bank(2) void print_matrix8(const Matrix8* m, char* name);

// Matrix32
__bank(2) Matrix32 init_m32(lsize_t width, lsize_t height);
__bank(2) void free_m32(Matrix32* m);
__bank(2) void print_matrix32(const Matrix32* m, char* name);

// Vector8
__bank(2) Vector8 init_v8(lsize_t length);
__bank(2) void free_v8(Vector8* v);
__bank(2) void print_vector8(const Vector8* v, char* name);

// Matrix Operations
__bank(2) void mul8(const Matrix8* A, const Matrix8* B, Matrix32* C);
__bank(2) Matrix32 get_mul8(const Matrix8* A, const Matrix8* B);
__bank(2) Matrix32 get_mul32(const Matrix32* A, const Matrix8* B);
__bank(2) Matrix32 to_mat32(const Matrix8* m);
__bank(2) void sub8(const Matrix8* M, const Matrix8* sub);
__bank(2) void mul8_1t(const Matrix8* A_T, const Matrix8* B, Matrix32* C);
__bank(2) Matrix32 get_mul8_1t(const Matrix8* A_T, const Matrix8* B);
__bank(2) void mul8_2t(const Matrix8* A, const Matrix8* B_T, Matrix32* C);
__bank(2) Matrix32 get_mul8_2t(const Matrix8* A, const Matrix8* B_T);

// Other
__bank(2) uint8_t ceil_log_2(int32_t value);
__bank(2) void relu8(const Matrix8* m);
__bank(2) uint8_t effective_bitwidth(const Matrix32* matrix);

__bank(2) void lin_cleanup();

#endif