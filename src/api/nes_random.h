#ifndef NES_RANDOM
#define NES_RANDOM

#include <stdint.h>

__attribute__((weak)) uint16_t nes_rand(void);
__attribute__((weak)) void nes_srand(uint16_t s);
__attribute__((weak)) uint32_t nes_rand_32(void);
__attribute__((weak)) void nes_srand_32(uint32_t s);

// By default, use modulo + tempering with no multiplications
// They say, higher bits have better statistics, but NES has no constant-time bit shifting
// All observations and research can be found in run-logs/plots/plot-analyze.ipynb
#ifndef RANDOM_TEMPERING_NO_MUL
#define RANDOM_TEMPERING_NO_MUL
#endif
#if !defined (RANDOM_RS) && !defined (RANDOM_LOW_BITS) && !defined (RANDOM_GLIBC)
#define RANDOM_LOW_BITS
#endif

#define RANDOM_32_MAX 0xFFFFFFFF

#define rand32() nes_rand_32()

// Turns out this method is useless
#ifdef RANDOM_RS
// ——— Rejection sampling ———
// вместо “% 256” и “% 65536”:
static inline uint8_t urand8_rs(void) {
    const uint32_t bound = 256;
    const uint32_t lim   = UINT32_MAX - (UINT32_MAX % bound);
    uint32_t r;
    do {
        r = nes_rand_32();
    } while (r >= lim);
    return (uint8_t)(r & 0xFF);
}
static inline uint16_t urand16_rs(void) {
    const uint32_t bound = 65536;
    const uint32_t lim   = UINT32_MAX - (UINT32_MAX % bound);
    uint32_t r;
    do {
        r = nes_rand_32();
    } while (r >= lim);
    return (uint16_t)(r & 0xFFFF);
}

#define urand8() urand8_rs()
#define urand16() urand16_rs()
#elif defined(RANDOM_LOW_BITS)
#define urand16() ((uint16_t)((nes_rand_32() >> 16) & 0xFFFF))
#define urand8() ((uint8_t) ((nes_rand_32() >> 24) & 0xFF))
#elif defined(RANDOM_GLIBC)
#include <stdlib.h>
#define urand8() ((uint8_t) (rand() & 0xFF))
#define urand16() ((uint16_t) (rand() & 0xFFFF))
#undef rand32
#define rand32() rand()
#else
#define urand8() ((uint8_t) (nes_rand_32() & 0xFF))
#define urand16() ((uint16_t) (nes_rand_32() & 0xFFFF))
#endif

#define rand8() ((int8_t) (urand8() - 128))
#define rand16() ((int16_t) (urand16() - 32768))

#define srand8(seed) nes_srand(seed)
#define srand32(seed) nes_srand_32(seed)

#endif