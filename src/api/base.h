#ifndef BASE_NN_H
#define BASE_NN_H

#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <inttypes.h>

#define FMT_8 "%" PRId8
#define FMT_u8 "%" PRIu8
#define FMT_16 "%" PRId16
#define FMT_u16 "%" PRIu16
#define FMT_32 "%" PRId32
#define FMT_u32 "%" PRIu32
#define FMT_SIZE "%zu"
#define FMT_POINTER "%p"

#ifdef NES
#define FMT_LSIZE FMT_u8
#else
#define FMT_LSIZE FMT_u32
#endif

#ifdef __NES__
#include "brk.h"
// #include "pointer.h"
#include "my_print.h"
#include <neslib.h>
#define printf(fmt, ...) my_printf(fmt, ##__VA_ARGS__)
#else
#include <stdio.h>
#endif

// #include "random.h"
#if defined(__NES__) || defined(NEW_RANDOM)

// #define urand8() lfsr8_step()
// #define urand16() lfsr16_step()
#else
#endif
#include "nes_random.h"

// #define urand8() lfsr8_step()
// #define urand16() lfsr16_step()

#if defined(NES) || defined(__NES__)
#define print_counter(name)
#define DEBUG_LOG_LEVEL 0
// Linear size
typedef uint8_t lsize_t;
#else
#define print_counter(name) static int count_##name = 0; log(#name ": %d", ++count_##name);
#define DEBUG_LOG_LEVEL 1 // 0 = Off, 1 = Basic Logs, 2 = Verbose Logs
typedef uint32_t lsize_t;
#endif

#ifdef NO_PRINT
#define print(...)
#define print_num(num)
#define print_pair(n1, n2)
#define println(...)
#define fprintf(out, fmt ...)
#define printf(...)
#else
#define print(fmt, ...) printf(fmt, ##__VA_ARGS__)
#define print_num(num) print("%d\n", num)
#define print_pair(n1, n2) print("(%d, %d)\n", n1, n2)
#define println(fmt, ...) print(fmt "\n", ##__VA_ARGS__)
#define LOG_LIST_CAPACITY 100
#define PRINT_ALLOWED 1
#endif

#ifndef NO_PRINT
    #define fatal(fmt, ...) fprintf(stderr, "FATAL [%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); exit(1)
    #define assert_fatal(cond, fmt, ...) if (!(cond)) {fatal(fmt, ##__VA_ARGS__);}
#else
    #define fatal(...)
    #define assert_fatal(...)
#endif

#if !defined (NO_PRINT) && DEBUG_LOG_LEVEL >= 1
    #define log(fmt, ...) fprintf(stderr, "INFO [%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
    #define log(...)
#endif

#if !defined (NO_PRINT) && DEBUG_LOG_LEVEL >= 2
    #define debug(fmt, ...) fprintf(stderr, "DEBUG [%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
    #define debug(...)
#endif

#if !defined (NO_PRINT) && defined (EXIT_ON_ERROR)
#define error(fmt, ...) fprintf(stderr, "ERROR [%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); exit(1)
#elif !defined (NO_PRINT)
#define error(fmt, ...) fprintf(stderr, "ERROR [%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
#define error(...)
#endif

#define STR_(x) #x
#define GET_INT_STR(x) STR_(x)

#if DEBUG_LOG_LEVEL > 0 && defined(LINEAR_METRICS)

void insert(const char key[], int value);
int get(const char key[]);
int getIndex(const char key[]);
void printMap();

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

#ifdef VBCC
#undef println
#define println(...)
#undef print
#define print(...)
#undef fatal
#define fatal(...)
#else
#define __bank(bank)
#endif

__bank(2) void print_metrics();

__bank(2) void print_heap_bounds(void);

void print_heap_stats(void);

#endif