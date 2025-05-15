#ifndef BRK
#define BRK

#include <stddef.h>

#define malloc(size) assert_malloc(size)

void* assert_malloc(size_t val);

#endif