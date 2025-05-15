#ifndef MY_PRINTF_H
#define MY_PRINTF_H

#include <stdarg.h>

#define fprintf(std, fmt, ...) printf(fmt, ##__VA_ARGS__)

void my_printf(const char *format, ...);

#endif
