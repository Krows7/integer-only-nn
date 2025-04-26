#ifndef BASE_NN_H
#define BASE_NN_H

#include <stdlib.h>
#include <stdint.h>
// #include <math.h>

// #undef NES

// #define NES

// #ifdef NES
// #undef NES
// #endif

#ifdef NES
#define urand8() rand8()
// #define urand16() rand16()
#define urand16() rand() % 65536
// Linear size
typedef uint8_t lsize_t;
#define DEBUG_LOG_LEVEL 0
#define PRINT_ALLOWED 0
#define print(...)
#define print_num(num)
#define print_pair(n1, n2)
#define println(...)
// #define fprintf(out, fmt ...)
#define print_counter(name)
#else
#define urand8() rand() % 256
#define urand16() rand() % 65536
typedef uint32_t lsize_t;
#define DEBUG_LOG_LEVEL 1 // 0 = Off, 1 = Basic Logs, 2 = Verbose Logs
#define PRINT_ALLOWED 1
#define print(fmt, ...) printf(fmt, ##__VA_ARGS__)
#define print_num(num) print("%d\n", num)
#define print_pair(n1, n2) print("(%d, %d)\n", n1, n2)
#define println(fmt, ...) print(fmt "\n", ##__VA_ARGS__)
#define print_counter(name) static int count_##name = 0; log(#name ": %d", ++count_##name);
#define LOG_LIST_CAPACITY 100
#endif

#define rand8() urand8() - 128
#define rand16() urand16() - 32768

#define ASSERT_CHECK
#define EXIT_ON_ERROR 1

#if PRINT_ALLOWED > 0
    #define fatal(fmt, ...) fprintf(stderr, "FATAL [%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); exit(1)
    #define assert_fatal(cond, fmt, ...) if (!(cond)) {fatal(fmt, ##__VA_ARGS__);}
#else
    #define fatal(...)
    #define assert_fatal(...)
#endif

#if PRINT_ALLOWED > 0 && DEBUG_LOG_LEVEL >= 1
    #define log(fmt, ...) fprintf(stderr, "INFO [%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
    #define log(...)
#endif

#if PRINT_ALLOWED > 0 && DEBUG_LOG_LEVEL >= 2
    #define debug(fmt, ...) fprintf(stderr, "DEBUG [%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
    #define debug(...)
#endif


#if PRINT_ALLOWED > 0 && EXIT_ON_ERROR > 0
#define error(fmt, ...) fprintf(stderr, "ERROR [%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); exit(1)
#elif PRINT_ALLOWED > 0
#define error(fmt, ...) fprintf(stderr, "ERROR [%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
#define error(...)
#endif


#define STR_(x) #x
#define GET_INT_STR(x) STR_(x)

void insert(const char key[], int value);
int get(const char key[]);
int getIndex(const char key[]);
void printMap();

#define LINEAR_METRICS 1
#define ALLOW_LINEAR_VERBOSE DEBUG_LOG_LEVEL > 0 && LINEAR_METRICS > 0

#if ALLOW_LINEAR_VERBOSE > 0

typedef struct {
    uint32_t** counters;
    char** keys;
    lsize_t size;
    lsize_t capacity;
} LogList;

extern LogList log_list;

void add_log_entry(char key[], uint32_t* value);

#define VERBOSE_NAME STR_(d_get_mul8_2t)
#define lin_count(name) static uint32_t debug_##name = 0;
#define count_name(name) debug_##name
#define count_inc(name) \
static uint32_t count_name(name) = 0; \
if (count_name(name) == 0) add_log_entry(#name, &count_name(name)); \
count_name(name)++

#define print_log_entries() \
for (lsize_t i = 0; i < log_list.size; i++) { \
    log("%s: %d", log_list.keys[i], *log_list.counters[i]); \
}

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

void print_metrics();

#define STR_(x) #x
#define GET_INT_STR(x) STR_(x)

#define stub(...) \
true_def(__VA_ARGS__); \
insert(__FILE__ ":" GET_INT_STR(__LINE__), get(__FILE__ ":" GET_INT_STR(__LINE__)) + 1)

#define stub_1(...) \
true_def(__VA_ARGS__); \
insert(__FILE__ ":" GET_INT_STR(__LINE__), get(__FILE__ ":" GET_INT_STR(__LINE__)) + 1)

#ifdef VBCC
#undef println
#define println(...)
#undef print
#define print(...)
#undef fatal
#define fatal(...)
#endif

#endif