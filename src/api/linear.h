#ifndef NN_MATH_H
#define NN_MATH_H

#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#define DEBUG_LOG_LEVEL 1 // 0 = Off, 1 = Basic Logs, 2 = Verbose Logs

// Linear size
typedef uint32_t lsize_t;


#define PRINT_ALLOWED 1
#define ASSERT_CHECK
#define EXIT_ON_ERROR 1

#if PRINT_ALLOWED > 0 && DEBUG_LOG_LEVEL >= 1
    #define log(fmt, ...) fprintf(stderr, "INFO [%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
    #define log(fmt, ...)
#endif

#if PRINT_ALLOWED > 0 && DEBUG_LOG_LEVEL >= 2
    #define debug(fmt, ...) fprintf(stderr, "DEBUG [%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
    #define debug(fmt, ...)
#endif


#if PRINT_ALLOWED > 0 && EXIT_ON_ERROR > 0
#define error(fmt, ...) fprintf(stderr, "ERROR [%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); exit(1)
#elif PRINT_ALLOWED > 0
#define error(fmt, ...) fprintf(stderr, "ERROR [%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
#define error(fmt, ...)
#endif


#define STR_(x) #x
#define GET_INT_STR(x) STR_(x)

void insert(const char key[], int value);
int get(const char key[]);
int getIndex(const char key[]);
void printMap();

#define print_counter(name) static int count_##name = 0; log(#name ": %d", ++count_##name);

#define LINEAR_METRICS 1
#define ALLOW_LINEAR_VERBOSE DEBUG_LOG_LEVEL > 0 && LINEAR_METRICS > 0

#if ALLOW_LINEAR_VERBOSE > 0
#define VERBOSE_NAME STR_(d_get_mul8_2t)
#define lin_count(name) static uint32_t debug_##name = 0;
#define count_name(name) debug_##name
#define count_inc(name) debug_##name++
#define print_count(name) \
log("%s: %d", #name, count_name(name)); \
if (strcmp(VERBOSE_NAME, #name) == 0) { \
    log(#name " map:"); \
    printMap(); \
}
// #define true_def init_m32_
// #define init_m32_ stub
// #define true_def get_mul8_2t
// #define get_mul8_2t stub
#else
#define lin_count(name)
#define count_name(name)
#define count_inc(name)
#define print_count(name)
#endif

#define max(a, b) (a > b ? a : b)
#define min(a, b) (a < b ? a : b)
#define abs(value) (value < 0 ? - value : value)
#define sign(value) ((value > 0) - (value < 0))

#define print printf
#define print_num(num) printf("%d\n", num)
#define print_pair(n1, n2) printf("(%d, %d)\n", n1, n2)
#define println() printf("\n")

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
    int64_t** matrix;
    lsize_t width;
    lsize_t height;
    int8_t scale;
} Matrix64;

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

void init_pools();

// If there's no room in Pool32, calls init_m32
Matrix32* request_m32(lsize_t width, lsize_t height);
void release_m32(Matrix32* m);

// ------------ Function Prototypes ------------

// Matrix8
Matrix8 init_m8(lsize_t width, lsize_t height);
void free_m8(Matrix8* m);
void print_matrix8(const Matrix8* m, char* name);

// Matrix32
Matrix32 init_m32(lsize_t width, lsize_t height);
void free_m32(Matrix32* m);
void print_matrix32(const Matrix32* m, char* name);

// Matrix64
Matrix64 init_m64(lsize_t width, lsize_t height);
void free_m64(Matrix64* m);
void print_matrix64(const Matrix64* m, char* name);

// Vector8
Vector8 init_v8(lsize_t length);
void free_v8(Vector8* v);
void print_vector8(const Vector8* v, char* name);

// Matrix Operations
void mul8(const Matrix8* A, const Matrix8* B, Matrix32* C);
Matrix32 get_mul8(const Matrix8* A, const Matrix8* B);
Matrix32 get_mul32(const Matrix32* A, const Matrix8* B);
Matrix32 to_mat32(const Matrix8* m);
void sub8(const Matrix8* M, const Matrix8* sub);
void mul8_1t(const Matrix8* A_T, const Matrix8* B, Matrix32* C);
Matrix32 get_mul8_1t(const Matrix8* A_T, const Matrix8* B);
void mul8_2t(const Matrix8* A, const Matrix8* B_T, Matrix32* C);
Matrix32 get_mul8_2t(const Matrix8* A, const Matrix8* B_T);

// extern Matrix32** pool;

// Other
uint8_t ceil_log_2(int32_t value);
void relu8(const Matrix8* m);
uint8_t effective_bitwidth(const Matrix32* matrix);

void print_metrics();

void lin_cleanup();

#define STR_(x) #x
#define GET_INT_STR(x) STR_(x)

#define stub(...) \
true_def(__VA_ARGS__); \
insert(__FILE__ ":" GET_INT_STR(__LINE__), get(__FILE__ ":" GET_INT_STR(__LINE__)) + 1)

#define stub_1(...) \
true_def(__VA_ARGS__); \
insert(__FILE__ ":" GET_INT_STR(__LINE__), get(__FILE__ ":" GET_INT_STR(__LINE__)) + 1)

#endif // NN_MATH_H