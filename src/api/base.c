#include "base.h"

#if DEBUG_LOG_LEVEL > 0 && defined(LINEAR_METRICS)

LogList log_list = {(uint32_t*[LOG_LIST_CAPACITY]){ NULL }, (char*[LOG_LIST_CAPACITY]){ NULL }, 0, LOG_LIST_CAPACITY};

__bank(2) void add_log_entry(char key[], uint32_t* value) {
    if (log_list.size == log_list.capacity) {
        error("LogList capacity reached");
    }
    log_list.keys[log_list.size] = key;
    log_list.counters[log_list.size++] = value;
}

#endif

__bank(2) void print_metrics() {
    #if ALLOW_LINEAR_VERBOSE > 0
    log("Linear Math Metrics:");
    print_log_entries();
    #endif
}

#ifdef VBCC
extern char __heap;
extern char __heapend;

__bank(2) void print_heap_bounds(void) {
    printf("Heap start @ %p\n", &__heap);
    printf("Heap   end @ %p\n", &__heapend);
}
#else
__bank(2) void print_heap_bounds(void) {
}
#endif

#ifdef __NES__
#include <stdlib.h>

extern unsigned char __heap_start;
extern size_t __stack;

void print_heap_stats(void) {
    // println("Heap usage: %zu/%zu bytes (%zu left)", __heap_bytes_used(), __heap_limit(), __heap_bytes_free());
    // println("Heap start: %zu, Stack: %zu", &__heap_start, __stack);
    // println("Heap start: %zu, Stack: %zu", &__heap_start, nes_get_soft_sp());
}
#else
void print_heap_stats(void) {
}
#endif
