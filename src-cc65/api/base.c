#include "base.h"
#include <stdio.h>

#ifndef NES

LogList log_list = {(uint32_t*[LOG_LIST_CAPACITY]){ NULL }, (char*[LOG_LIST_CAPACITY]){ NULL }, 0, LOG_LIST_CAPACITY};

void add_log_entry(char key[], uint32_t* value) {
    if (log_list.size == log_list.capacity) {
        error("LogList capacity reached");
    }
    log_list.keys[log_list.size] = key;
    log_list.counters[log_list.size++] = value;
}

#endif

void print_metrics() {
    #if ALLOW_LINEAR_VERBOSE > 0
    log("Linear Math Metrics:");
    print_log_entries();
    #endif
}