#ifndef QUANTIZATION_H
#define QUANTIZATION_H

#include "linear.h"

#define BITWIDTH 7 // Target bitwidth for quantization

// RoundShift (deterministic)
int8_t round_shift(int32_t input, int8_t shift);

// PstoShift (pseudo-stochastic)
int8_t psto_shift(int32_t input, int8_t shift);

// --- Helper Functions ---
int8_t get_int32_bitwidth(int32_t val); // Estimate bitwidth of a single int32
int8_t estimate_matrix_bitwidth(Matrix32* m); // Estimate max bitwidth in a 32-bit matrix

#endif