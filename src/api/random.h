#ifndef RANDOM_H
#define RANDOM_H

#include <stdint.h>

/*——————————————————————————————————————————————————————————————————————
  — 8-bit LFSR: polynomial x^8 + x^6 + x^5 + x^4 + 1 (0xB8)
  — period = 2^8–1 = 255 (state must never be 0)
  ——————————————————————————————————————————————————————————————————————*/

__attribute__((leaf)) char rand8(void);

__attribute__((leaf)) unsigned rand16(void);

uint8_t urand8(void);

uint16_t urand16(void);

uint8_t lfsr8_step(void);

uint16_t lfsr16_step(void);


// static uint8_t lfsr8_state = 1u;

// static inline uint8_t lfsr8_step(void) {
//     /* grab low bit for feedback */
//     uint8_t lsb = lfsr8_state & 1u;
//     /* shift right by 1 */
//     lfsr8_state >>= 1;
//     /* if bit0 was 1, apply tap mask */
//     if (lsb) lfsr8_state ^= 0xB8u;
//     return lfsr8_state;
// }

// /*——————————————————————————————————————————————————————————————————————
//   — 16-bit LFSR: polynomial x^16 + x^14 + x^13 + x^11 + 1 (0xB400)
//   — period = 2^16–1 = 65535 (state must never be 0)
//   ——————————————————————————————————————————————————————————————————————*/
// static uint16_t lfsr16_state = 1u;

// static inline uint16_t lfsr16_step(void) {
//     uint16_t lsb = lfsr16_state & 1u;
//     lfsr16_state >>= 1;
//     if (lsb) lfsr16_state ^= 0xB400u;
//     return lfsr16_state;
// }

// /*——————————————————————————————————————————————————————————————————————
//   — Public APIs
//   ——————————————————————————————————————————————————————————————————————*/

// /// unsigned 8-bit [1..255]
// static inline uint8_t urand8(void)   { return lfsr8_step(); }

// /// signed 8-bit  (casts 0..255 → −128..127)
// static inline int8_t  rand8(void)    { return (int8_t)lfsr8_step(); }

// /// unsigned 16-bit [1..65535]
// static inline uint16_t urand16(void) { return lfsr16_step(); }

// /// signed 16-bit (casts 0..65535 → −32768..32767)
// static inline int16_t  rand16(void)  { return (int16_t)lfsr16_step(); }

// /*——————————————————————————————————————————————————————————————————————
//   — Optional: allow the user to seed; must not set state to 0
//   ——————————————————————————————————————————————————————————————————————*/
// static inline void srand8(uint8_t seed) {
//     if (seed == 0) seed = 1;
//     lfsr8_state = seed;
// }

// static inline void srand16(uint16_t seed) {
//     if (seed == 0) seed = 1;
//     lfsr16_state = seed;
// }

#endif // RANDOM_H