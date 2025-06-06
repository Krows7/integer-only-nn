// my_printf.c (in your project)
#include "my_print.h" // Your header defining my_printf
#include <stdarg.h>
#include <stdint.h>   // For uintptr_t, intmax_t, etc.
#include "stddef.h"
// #include <stdio.h> // For putchar, if it's safe. Otherwise, implement custom char output.
// You'll need a way to output single characters. Let's assume putchar() is available and works.
// If not, you'll need to replace putchar() with your target's specific char output routine.
#ifdef __NES__ // Assuming putchar is available from SDK and is safe enough for this.
#include "stdio.h"
#else
// extern void my_custom_putchar(char c); // Your custom char output
// #define putchar my_custom_putchar
#endif


#ifdef __NES__
// Helper to reverse a string
static void reverse_str(char* str, int length) {
    int start = 0;
    int end = length - 1;
    char temp;
    while (start < end) {
        temp = str[start];
        str[start] = str[end];
        str[end] = temp;
        start++;
        end--;
    }
}

// Custom unsigned long long to string (supports different bases)
// static char* ull_to_str(unsigned long long num, char* buffer, int base, int is_uppercase) {
//     int i = 0;
//     if (num == 0) {
//         buffer[i++] = '0';
//         buffer[i] = '\0';
//         return buffer;
//     }

//     while (num != 0) {
//         int rem = num % base;
//         if (rem < 10) {
//             buffer[i++] = rem + '0';
//         } else {
//             buffer[i++] = (is_uppercase ? 'A' : 'a') + (rem - 10);
//         }
//         num = num / base;
//     }
//     buffer[i] = '\0';
//     reverse_str(buffer, i);
//     return buffer;
// }

const int32_t pow10[] = {1000000000,100000000,10000000,1000000,100000,10000,1000,100,10,1};

// Assume base == 10
static uint8_t ull_to_str(uint32_t num, char* buffer, uint8_t base, uint8_t is_uppercase) {
    if (num == 0) {
        buffer[0] = '0';
        buffer[1] = '\0';
        return 1;
    }
    uint8_t i = 0;
    uint8_t started = 0;
    for (uint8_t j = 0; j < 10; ++j) {
        uint8_t digit = 0;
        while (num >= pow10[j]) {
            num -= pow10[j];
            ++digit;
        }
        if (digit > 0 || started || j == 9) {
            buffer[i++] = '0' + digit;
            started = 1;
        }
    }

    // while (num != 0) {
    //     int rem = num % base;
    //     buffer[i++] = rem + '0';
    //     num /= base;
    // }
    buffer[i] = '\0';
    // reverse_str(buffer, i);
    return i;
}

// Custom long long to string (base 10 only for simplicity here)
uint8_t ll_to_str(int32_t num, char* buffer) {
    unsigned long long n_val;
    uint8_t is_negative = 0;

    if (num == 0) {
        buffer[0] = '0';
        buffer[1] = '\0';
        return 1;
        // return buffer;
    }

    if (num < 0) {
        is_negative = 1;
        // Handle LLONG_MIN carefully if it's a concern: -(LLONG_MIN) can overflow
        // if (num == -9223372036854775807LL - 1) { // LLONG_MIN
        //      n_val = 9223372036854775807LL; // LLONG_MAX
        //      // Special handling for LLONG_MIN if needed, or print a fixed string
        // } else {
            n_val = -num;
        // }
    } else {
        n_val = num;
    }
    
    // Use ull_to_str for the positive part
    if (is_negative) {
        buffer[0] = '-';
    }
    return is_negative + ull_to_str(n_val, buffer + is_negative, 10, 0); // Write after potential sign
    // return buffer;
}

static void my_puts_minimal(const char *s) {
    if (!s) s = "(null)";
    while(*s) {
        putchar(*s++);
    }
}
// char num_buffer[65]; // Max for 64-bit number in binary + null

void my_printf(const char *format, ...) {
    char num_buffer[11];
    // my_puts_minimal("A");
    va_list args;
    va_start(args, format);
    const char* p = format;

    while (*p != '\0') {
        if (*p == '%') {
            p++;
            if (*p == '\0') { putchar('%'); break; }

            // Simplified: No flags, width, precision parsing for this example
            
            char length_modifier = 0; // 0:none, 'h':short, 'H':char, 'l':long, 'L':long long, 'z':size_t
            if (*p == 'l') {
                p++;
                if (*p == 'l') { length_modifier = 'L'; p++; } // ll
                else { length_modifier = 'l'; } // l
            } else if (*p == 'h') {
                p++;
                if (*p == 'h') { length_modifier = 'H'; p++; } // hh
                else { length_modifier = 'h'; } // h
            } else if (*p == 'z') {
                length_modifier = 'z'; p++;
            }

            char specifier = *p;
            int is_uppercase_hex = (specifier == 'X');

            switch (specifier) {
                case 'd':
                case 'i': {
                    int32_t val;
                    if (length_modifier == 'L') val = va_arg(args, long long);
                    else if (length_modifier == 'l') val = va_arg(args, long);
                    else if (length_modifier == 'z') val = (long long)va_arg(args, ptrdiff_t); // for %zd
                    else val = va_arg(args, int); // char, short, int
                    // Apply 'h'/'H' if needed by casting val before ll_to_str
                    if (length_modifier == 'h') val = (short)val;
                    else if (length_modifier == 'H') val = (signed char)val;
                    ll_to_str(val, num_buffer);
                    my_puts_minimal(num_buffer);
                    break;
                }
                case 'u':
                case 'x':
                case 'X': {
                    uint32_t val;
                    int base = 10;
                    // if (specifier == 'x' || specifier == 'X') base = 16;

                    if (length_modifier == 'L') val = va_arg(args, unsigned long long);
                    else if (length_modifier == 'l') val = va_arg(args, unsigned long);
                    else if (length_modifier == 'z') val = (unsigned long long)va_arg(args, size_t); // for %zu, %zx
                    else val = va_arg(args, unsigned int); // char, short, int
                    
                    if (length_modifier == 'h') val = (unsigned short)val;
                    else if (length_modifier == 'H') val = (unsigned char)val;
                    
                    ull_to_str(val, num_buffer, base, is_uppercase_hex);
                    my_puts_minimal(num_buffer);
                    break;
                }
                // case 'p': {
                //     uintptr_t ptr_val = (uintptr_t)va_arg(args, void*);
                //     my_puts_minimal("0x"); // Optional prefix
                //     ull_to_str((unsigned long long)ptr_val, num_buffer, 16, 0); // Pointers as lowercase hex
                //     my_puts_minimal(num_buffer);
                //     break;
                // }
                case 's': {
                    char *s_val = va_arg(args, char *);
                    my_puts_minimal(s_val);
                    break;
                }
                // case 'c': {
                //     char c_val = (char)va_arg(args, int); // char promotes to int
                //     putchar(c_val);
                //     break;
                // }
                case '%': {
                    putchar('%');
                    break;
                }
                default: // Unsupported specifier
                    putchar('%');
                    // Print length modifier if present
                    if (length_modifier == 'L') { putchar('l'); putchar('l'); }
                    else if (length_modifier == 'l') putchar('l');
                    else if (length_modifier == 'H') { putchar('h'); putchar('h'); }
                    else if (length_modifier == 'h') putchar('h');
                    else if (length_modifier == 'z') putchar('z');
                    putchar(specifier); // The unrecognized specifier character
                    break;
            }
        } else {
            putchar(*p);
        }
        p++;
    }
    va_end(args);
}
#else
#endif