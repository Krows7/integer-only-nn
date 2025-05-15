#ifndef POINTER_H
#define POINTER_H

#include <stdint.h>

/* declare the imaginary rs0 register (soft-SP) */
// extern volatile uint8_t __rc0;
// extern volatile uint8_t __rc1;

// /* returns the current software stack pointer (0x0000–0xFFFF) */
// static inline uint16_t nes_get_soft_sp(void) {
//     uint16_t lo = __rc0;           // read low byte
//     uint16_t hi = __rc1;           // read high byte
//     return lo | (hi << 8);         // little-endian assembly
// }


static uint8_t* rc0 = (uint8_t*) 0x0;
static uint8_t* rc1 = (uint8_t*) 0x1;

/* returns the current software stack pointer (0x0000–0xFFFF) */
static inline uint16_t nes_get_soft_sp(void) {
    uint16_t lo = *rc0;           // read low byte
    uint16_t hi = *rc1;           // read high byte
    return lo | (hi << 8);         // little-endian assembly
}

#endif