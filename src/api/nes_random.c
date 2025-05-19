#include "nes_random.h"
#include <stdint.h>

// A 16-bit xorshift generator with constants well suited to 8-bit systems.
// Values found by George Marsaglia for the Z80:
//   http://www.retroprogramming.com/2017/07/xorshift-pseudorandom-numbers-in-z80.html

// Modified based on LLVM-MOS C stdlib.cc

static uint16_t seed = 1;

__attribute__((weak)) uint16_t nes_rand(void) {
  uint16_t x = seed;
  x ^= x << 7;
  x ^= x >> 9;
  x ^= x << 8;
  return seed = x;
}

__attribute__((weak)) void nes_srand(uint16_t s) { seed = s; }

static uint32_t seed32 = 1;

#ifdef RANDOM_TEMPERING

// ——— улучшенный xorshift32 с темперингом и Weyl ———

static uint32_t weyl   = 0;
// константа Weyl для 32-бит: 2^32 / φ ≈ 0x9E3779B9 
#define WEYL_CONST 0x9E3779B9u

// Темперинг “по мотивам” MurmurHash3 finalizer
static inline uint32_t temper(uint32_t x) {
    x ^= x >> 16;
    x *= 0x85EBCA6Bu;
    x ^= x >> 13;
    x *= 0xC2B2AE35u;
    x ^= x >> 16;
    return x;
}

__attribute__((weak)) uint32_t nes_rand_32(void) {
    // стандартный xorshift32
    uint32_t x = seed32;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    seed32 = x;
    // добавляем Weyl-сдвиг, чтобы разорвать циклические корреляции
    weyl   += WEYL_CONST;
    x     += weyl;
    // наконец, температурим для «хорошего» перемешивания битов
    return temper(x);
}

__attribute__((weak)) void nes_srand_32(uint32_t s) {
    seed32 = s;
    weyl    = 0;      // сбрасываем Weyl-счетчик
}
#elif defined(RANDOM_TEMPERING_NO_MUL)
static uint32_t weyl   = 0;

static inline uint32_t temper_no_mul(uint32_t x) {
    // Простой «битовый миксер» на основе xorshift-темперинга
    x ^= x >> 16;
    x ^= x << 10;
    x ^= x >>  4;
    x ^= x <<  5;
    x ^= x >> 15;
    return x;
}

// константа Weyl для 32-бит: 2^32 / φ ≈ 0x9E3779B9 
#define WEYL_CONST 0x9E3779B9u

__attribute__((weak)) uint32_t nes_rand_32(void) {
    // стандартный xorshift32
    uint32_t x = seed32;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    seed32 = x;
    // добавляем Weyl-сдвиг, чтобы разорвать циклические корреляции
    weyl   += WEYL_CONST;
    x     += weyl;
    // наконец, температурим для «хорошего» перемешивания битов
    return temper_no_mul(x);
}

__attribute__((weak)) void nes_srand_32(uint32_t s) {
    seed32 = s;
    weyl    = 0;      // сбрасываем Weyl-счетчик
}
#else
__attribute__((weak)) uint32_t nes_rand_32(void)
{
	/* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
	uint32_t x = seed32;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	return seed32 = x;
}

__attribute__((weak)) void nes_srand_32(uint32_t s) { seed32 = s; }
#endif