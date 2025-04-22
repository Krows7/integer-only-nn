#ifndef NN_MATH_H
#define NN_MATH_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define DEBUG

#ifdef DEBUG
    #define log(fmt, ...) fprintf(stderr, "DEBUG [%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
    #define log(fmt, ...) ;
#endif

#define max(a, b) (a > b ? a : b)
#define min(a, b) (a < b ? a : b)
#define abs(value) (value < 0 ? - value : value)
#define sign(value) ((value > 0) - (value < 0))

#define i88 uint8_t
#define uint8_t uint32_t

#define print printf
#define print_num(num) printf("%d\n", num)
#define print_pair(n1, n2) printf("(%d, %d)\n", n1, n2)
#define println() printf("\n")

// ------------ Struct Definitions ------------
typedef struct {
    int8_t** matrix;
    uint8_t width;
    uint8_t height;
    int8_t scale;
} Matrix8;

typedef struct {
    int32_t** matrix;
    uint8_t width;
    uint8_t height;
    int8_t scale;
} Matrix32;

typedef struct {
    int64_t** matrix;
    uint8_t width;
    uint8_t height;
    int8_t scale;
} Matrix64;

typedef struct {
    int8_t* vector;
    uint8_t length;
    int8_t scale;
} Vector8;

// ------------ Function Prototypes ------------

// Matrix8
Matrix8 init_m8(uint8_t width, uint8_t height);
void free_m8(Matrix8* m);
void print_matrix8(Matrix8* m, char* name);

// Matrix32
Matrix32 init_m32(uint8_t width, uint8_t height);
void free_m32(Matrix32* m);
void print_matrix32(Matrix32* m, char* name);

// Matrix64
Matrix64 init_m64(uint8_t width, uint8_t height);
void free_m64(Matrix64* m);
void print_matrix64(Matrix64* m, char* name);

// Vector8
Vector8 init_v8(uint8_t length);
void free_v8(Vector8* v);
void print_vector8(Vector8* v, char* name);

// Matrix Operations
void mul8(const Matrix8* A, const Matrix8* B, Matrix32* C);
Matrix32 get_mul8(const Matrix8* A, const Matrix8* B);
Matrix32 get_mul32(const Matrix32* A, const Matrix8* B);
Matrix32 to_mat32(Matrix8* m);
void sub8(Matrix8* M, const Matrix8* sub);
void mul8_1t(const Matrix8* A_T, const Matrix8* B, Matrix32* C);
Matrix32 get_mul8_1t(const Matrix8* A_T, const Matrix8* B);
void mul8_2t(const Matrix8* A, const Matrix8* B_T, Matrix32* C);
Matrix32 get_mul8_2t(const Matrix8* A, const Matrix8* B_T);

// Other
uint8_t ceil_log_2(int32_t value);
void relu8(Matrix8* m);
uint8_t effective_bitwidth(Matrix32* matrix);

#endif // NN_MATH_H