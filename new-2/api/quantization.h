#ifndef LINEAR_MATH_H // Or QUANTIZATION_H
#define LINEAR_MATH_H

#include "linear.h"

#define BITWIDTH 7 // Target bitwidth for quantization

// --- New Rounding Functions ---

// RoundShift (deterministic)
int8_t round_shift(int32_t input, int8_t shift);

// PstoShift (pseudo-stochastic)
int8_t psto_shift(int32_t input, int8_t shift);

// --- Helper Functions ---
int8_t get_int32_bitwidth(int32_t val); // Estimate bitwidth of a single int32
int8_t estimate_matrix_bitwidth(Matrix32* m); // Estimate max bitwidth in a 32-bit matrix

// --- Existing Function Declarations (Ensure they handle scale/exponents) ---
// ... (init_m8, free_m8, init_v8, free_v8, etc.) ...
// ... (matrix multiplication functions: get_mul8, get_mul8_1t, get_mul8_2t) ...
// Note: Matrix multiplication functions might need updates if they implicitly assumed scale=0

#endif // LINEAR_MATH_H