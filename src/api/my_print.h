#ifndef MY_PRINTF_H
#define MY_PRINTF_H

#include <stdarg.h>
#include <stdint.h>

#define fprintf(std, fmt, ...) printf(fmt, ##__VA_ARGS__)

void my_printf(const char *format, ...);

uint8_t ll_to_str(int32_t num, char* buffer);

#endif
