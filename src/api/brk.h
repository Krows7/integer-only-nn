#ifndef BRK
#define BRK

#include <stddef.h>

// #define malloc(size) assert_malloc(size)
#define malloc(size) assert_malloc_log(size, __FILE__, __LINE__)

void* assert_malloc(size_t val);

void* assert_malloc_log(size_t val, const char* file, int line);

#endif