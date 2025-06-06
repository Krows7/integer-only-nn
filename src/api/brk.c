#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include "brk.h"
#include "base.h"

__attribute__((section(".bss"))) static volatile uint16_t i = 0;

#ifdef malloc
#undef malloc
#endif

void write_word(size_t value) {
    (*(volatile size_t*)0x4018) = value;
    (*(volatile uint8_t*)0x401A) = 1;
}

#ifndef __NES__
#define write_word(value)
#endif

void* assert_malloc_log(size_t size, const char* file, int line) {
    // println("MALLOC [%s:%d] %zu", file, line, size);
    return assert_malloc(size);
}

extern size_t __heap_bytes_used();
extern size_t __heap_limit();
extern size_t __heap_bytes_free();

// void* assert_malloc(size_t size) {
void* assert_malloc(size_t val) {
    // if (!size) return NULL;
    ++i;
    // print_heap_stats();
    // printf("Call Before: %zu, Requested: %zu\n", i, val);
    // printf("Call Before:\n");
    // write_word(i);
    void* res = malloc(val);
    // printf("Call After:\n");
    write_word(i);
    // printf("Heap Bytes Used:\n");
    // write_word(__heap_bytes_used());
    // // printf("Heap Limit:\n");
    // write_word(__heap_limit());
    // // printf("Heap Bytes Free:\n");
    // write_word(__heap_bytes_free());
    // printf("Call After: %zu, Malloc*: %p\n", i, res);
    // void* res = (void*) 1;
    if (!res && val > 0) {
        // (*(size_t*)0x4018) = 122;
        write_word(122);
        // write_word(i);
        write_word(val);
        write_word(__heap_bytes_used());
        // printf("Heap Limit:\n");
        write_word(__heap_limit());
        // printf("Heap Bytes Free:\n");
        write_word(__heap_bytes_free());

        // fatal("assert_malloc: Malloc call #%u failed to allocate %zu bytes.", (unsigned int)i, val);
        // (*(size_t*)0x4018) = __heap_bytes_used();
        // (*(size_t*)0x4018) = __heap_limit();
        // (*(size_t*)0x4018) = __heap_bytes_free();
        // printf("Error at: %zu\n", i);
        // if (i == 0) {
        //     ++i;
        //     return val;
        // }
        // printf("AAAAAAAAA\n");
        // exit(1);
    }
    return res;
}