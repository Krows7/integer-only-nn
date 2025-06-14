#include <stdint.h>
#include "quantization.h"
#include "base.h"
#include "linear.h"

// --- Helper: Estimate bitwidth of a single int32 ---
// Returns the minimum bits needed to represent the value (excluding sign bit)
// Equivalent to ceil(log2(abs(val))) if val != 0
int8_t get_int32_bitwidth(int32_t val) {
    if (val == 0) return 0;
    // Use unsigned to handle INT_MIN correctly
    uint32_t uval = (val == INT32_MIN) ? (uint32_t)INT32_MAX + 1 : (uint32_t)abs(val);
    // Find position of the most significant bit (MSB)
    // __builtin_clz is efficient on GCC/Clang
    #if !defined(__NES__) && (defined(__GNUC__) || defined(__clang__))
        int leading_zeros = __builtin_clz(uval);
        return (32 - leading_zeros);
    #else
        int8_t bits = 0;
        while (uval > 0) {
            uval >>= 1;
            bits++;
        }
        return bits;
    #endif
}

// --- Helper: Estimate max bitwidth in a 32-bit matrix ---
int8_t estimate_matrix_bitwidth(Matrix32* m) {
    uint32_t max_abs_val = 0;
    for (size_t i = 0; i < m->width; ++i) {
        for (size_t j = 0; j < m->height; ++j) {
            int32_t current_val = m->matrix[i][j];
            uint32_t abs_val = (current_val == INT32_MIN) ? (uint32_t)INT32_MAX + 1 : (uint32_t)abs(current_val);
            if (abs_val > max_abs_val) { // Use unsigned comparison
                 max_abs_val = abs_val;
            }
        }
    }
    // print_matrix32(m, "Estimate");
    // println("Max: %ld", max_abs_val);
     // Directly calculate bitwidth from the max absolute value found
    return get_int32_bitwidth(max_abs_val);
}


// --- Rounding Functions ---

// RoundShift (deterministic rounding to nearest, ties to even - like Python round())
// Note: C's roundf rounds away from zero for .5, Python rounds to nearest even.
// Implementing Python's round-half-to-even is more complex.
// Let's use standard roundf for simplicity first.
// A simpler deterministic approach matching the Python code's apparent logic:
int8_t round_shift(int32_t input, int8_t shift) {
    if (shift <= 0) {
        // No shift or invalid shift, clamp and return
        if (input > 127) return 127;
        if (input < -128) return -128;
        return (int8_t)input;
    }
    // int32_t divisor = ((int32_t) 1) << shift;
    int32_t half_divisor = (int32_t) 1 << (shift - 1); // For rounding

    // println("AA: %d %ld %ld", shift, divisor, half_divisor);

    // Add half_divisor for positive numbers, subtract for negative before division
    int32_t rounded_val;
    if (input >= 0) {
        // rounded_val = (input + half_divisor) / divisor;
        rounded_val = (input + half_divisor) >> shift;
    } else {
        // Careful with integer division of negative numbers
        // rounded_val = (input - half_divisor + 1) / divisor; // Approximation
        // rounded_val = (input - half_divisor + 1) >> shift; // Approximation
        // int32_t true_round = (input - half_divisor + 1) / divisor;
        // if (true_round != rounded_val + 1) println(FMT_32 "==" FMT_32 " (" FMT_32 ", " FMT_32 ", " FMT_8 ", " FMT_32 ")", rounded_val, true_round, input, half_divisor, shift, divisor);
        // else println("AAA " FMT_32 "==" FMT_32 " (" FMT_32 ", " FMT_32 ", " FMT_8 ", " FMT_32 ")", rounded_val, true_round, input, half_divisor, shift, divisor);
        // A more robust way for negative numbers:
        // rounded_val = -((abs(input) + half_divisor) / divisor);

        rounded_val = -((half_divisor - 1 - input) >> shift);
    }

    // printf("Max stack used: %u bytes\n", (unsigned)__heap);

    // println("%ld %ld %ld", rounded_val, input + half_divisor, (input + half_divisor) >> shift);

    // Clamp
    if (rounded_val > 127) return 127;
    if (rounded_val < -128) return -128;
    return (int8_t)rounded_val;
}

#ifdef LIN_DEBUG
#define print_i8(fmt, num) println(fmt ": %d", num)
#ifndef __NES__
#define print_i32(fmt, num) println(fmt ": %d", num)
#else
#define print_i32(fmt, num) println(fmt ": %ld", num)
#endif
#else
#define print_i32(fmt, num)
#define print_i8(fmt, num)
#endif

// void print_i32(const char* fmt, int32_t num) {
//     #ifndef __NES__
//     println(fmt "%d", num);
//     #else
//     println("%ld", num);
//     #endif
// }

// PstoShift (Pseudo-stochastic rounding)
// shift must be non-negative
// input == INT32_MIN will provide errors
int8_t psto_shift(int32_t input, int8_t shift) {
     if (shift <= 0) {
        if (input > 127) return 127;
        if (input < -128) return -128;
        return (int8_t)input;
    }

    // int32_t divisor = 1 << shift;
    // int32_t divisor = ((int32_t) 1) << shift;
    // print_i32("psto divisor", divisor);

    // int32_t round_temp = input / divisor; // Integer division truncates
    // int32_t round_temp = input >> shift;
    int32_t round_temp;
    if (input >= 0) {
        round_temp = input >> shift;
    } else {
        round_temp = -(-input >> shift); // Integer division truncates
    }
    print_i32("psto round_temp", round_temp);

    // Calculate remainder (probability)
    // int32_t remainder = input - (round_temp << divisor);
    // int32_t remainder = input - round_temp * divisor;
    int32_t remainder = input - (round_temp << shift);
    print_i32("psto input", input);
    print_i32("psto interm", round_temp << shift);
    print_i32("psto remainder", remainder);

    uint32_t prob = abs(remainder); // Use absolute value for comparison magnitude

    // Calculate pseudo-random number from lower bits of remainder/prob
    // int8_t sub_shift = shift / 2;
    int8_t sub_shift = shift >> 1;
    print_i8("psto sub_shift", sub_shift);

    uint32_t pseudo_rand_num = prob & ((((int32_t) 1) << sub_shift) - 1); // Extract lower bits
    print_i32("psto pseudo_rand_num", pseudo_rand_num);

    // Quantize the probability (upper bits of prob)
    uint32_t quantized_prob = prob >> sub_shift;
    print_i32("psto quantized_prob", quantized_prob);
    

    // Adjust pseudo_rand_num if shift is odd (match bit widths)
    if (shift % 2 == 1) {
        pseudo_rand_num <<= 1;
    }
    print_i32("psto pseudo_rand_num", pseudo_rand_num);

    // Make rounding decision
    int8_t round_decision = (quantized_prob <= pseudo_rand_num) ? 0 : 1;

    // Apply sign
    if (input < 0) {
        round_decision = -round_decision;
    }

    int32_t final_val = round_temp + round_decision;
    print_i32("psto final_val", final_val);

    // Clamp
    if (final_val > 127) return 127;
    if (final_val < -128) return -128;
    return (int8_t)final_val;
}

// // Helper: Clip to int8 range [-127, 127] (matching Python's clip_val=127)
// // Note: Python clips to [-127, 127], not [-128, 127]. Let's match that.
// // If you need [-128, 127], adjust the min value.
// static inline int8_t int8_clip_c(int32_t input) {
//     if (input > 127) return 127;
//     if (input < -127) return -127; // Match Python's clip_val=127
//     return (int8_t)input;
// }

// // C equivalent of Python's RoundShift
// // Performs standard rounding: add half the divisor, then shift right (truncate).
// static inline int8_t round_shift_c(int32_t input, int shift) {
//     if (shift <= 0) {
//         // No shift or invalid shift, just clip
//         return int8_clip_c(input);
//     }
//     // Calculate half the divisor (avoiding potential overflow with large inputs)
//     int32_t half_divisor = (1 << (shift - 1));
//     int32_t rounded_input;

//     if (input >= 0) {
//         rounded_input = (input + half_divisor) >> shift;
//     } else {
//         // For negative numbers, standard rounding means adding half the divisor magnitude
//         // then shifting. Equivalently, subtract half before shifting.
//         // Example: -5 shift 1 -> (-5 + 1) >> 1 = -4 >> 1 = -2. Correct.
//         // Example: -3 shift 1 -> (-3 + 1) >> 1 = -2 >> 1 = -1. Correct.
//         // Example: -4 shift 1 -> (-4 + 1) >> 1 = -3 >> 1 = -2 (truncation towards -inf). Correct.
//          rounded_input = (input + half_divisor) >> shift;
//         // Alternative for rounding away from zero for negatives: (input - half_divisor) >> shift
//         // But Python's RoundShift appears to round half towards +infinity, so the above is likely correct.
//     }

//     return int8_clip_c(rounded_input);
// }

// // C equivalent of Python's PstoShift (Pseudo-Stochastic Rounding)
// // Note: This directly translates the Python logic. Assumes int32 inputs.
// static inline int8_t psto_shift_c(int32_t input, int shift) {
//     if (shift <= 0) {
//         // No shift or invalid shift, just clip
//         // Python's grad_calc returns input.type(torch.int8) if shift < 1
//         // Let's return clipped input if shift <= 0 for safety.
//          return int8_clip_c(input);
//     }
//      if (input == 0) {
//          return 0;
//      }

//     int32_t divisor = (1 << shift);
//     int32_t round_temp = input / divisor; // Integer division truncates towards zero

//     // Calculate absolute remainder (prob)
//     int32_t prob = abs(input - round_temp * divisor);

//     // Calculate shift for quantization and pseudo-random number
//     int shift_quant = shift / 2; // Integer division
//     int shift_prand = shift - shift_quant; // shift_prand >= shift_quant

//     int32_t divisor_quant = (1 << shift_quant);
//     int32_t quantized_prob = prob / divisor_quant;

//     int32_t divisor_prand = (1 << shift_prand);
//     int32_t pseudo_rand_num = prob % divisor_prand; // Remainder after dividing by 2^shift_prand

//     // Python code: if shift is odd, need to make sure qprob and prand have same bit width
//     // if shift % 2 == 1: pseudo_rand_num = pseudo_rand_num*2
//     // This seems counter-intuitive. Let's re-evaluate the Python logic.
//     // It seems `pseudo_rand_num = prob - quantized_prob*(2**(shift//2))` is used.
//     // Let's stick to the direct translation first.
//     // `pseudo_rand_num = prob - quantized_prob * divisor_quant;` is equivalent to `prob % divisor_quant` if shift_prand == shift_quant.
//     // Let's recalculate pseudo_rand_num as per Python:
//     pseudo_rand_num = prob - quantized_prob * divisor_quant;

//     // Python: if shift % 2 == 1: pseudo_rand_num = pseudo_rand_num*2
//     // This adjustment seems necessary to compare fairly if bit widths differ.
//     if (shift % 2 == 1) {
//         pseudo_rand_num = pseudo_rand_num * 2;
//     }

//     // Round decision based on comparison
//     int32_t round_decision_val = (quantized_prob <= pseudo_rand_num) ? 0 : 1;

//     // Apply sign
//     int32_t sign = (input > 0) - (input < 0); // Gets 1, -1 or 0
//     round_decision_val = round_decision_val * sign;

//     // Final result
//     int32_t result = round_temp + round_decision_val;

//     return int8_clip_c(result);
// }

// // --- You might also need a C equivalent of RangeEstimate ---
// // This is crucial for determining the 'shift' value dynamically.

// // C equivalent of Python's RangeEstimate
// // Returns the minimum bitwidth needed to represent the max absolute value.
// // Returns 0 if max absolute value is 0.
// static inline int range_estimate_c(const int32_t* data, size_t num_elements) {
//     if (data == NULL || num_elements == 0) {
//         return 0; // Or handle error appropriately
//     }
//     int32_t max_abs_val = 0;
//     for (size_t i = 0; i < num_elements; ++i) {
//         int32_t abs_val = abs(data[i]);
//         if (abs_val > max_abs_val) {
//             max_abs_val = abs_val;
//         }
//     }

//     if (max_abs_val == 0) {
//         return 0;
//     }

//     // Calculate bits needed: ceil(log2(max_abs_val + 1)) if considering zero,
//     // or ceil(log2(max_abs_val)) if just the max value. Python uses ceil(log2(range)).
//     // Let's use log2. Requires linking with -lm.
//     // Add a small epsilon to handle exact powers of 2 correctly with floating point log.
//     // Or use integer methods.
//     // Integer method: find the position of the most significant bit.
//     if (max_abs_val <= 0) return 0; // Should be handled by max_abs_val == 0 check

//     int bits = 0;
//     // Find the highest set bit position (0-indexed)
//     // e.g., 127 (0111 1111) -> highest bit at 6. Needs 7 bits.
//     // e.g., 128 (1000 0000) -> highest bit at 7. Needs 8 bits.
//     // So, position + 1 is the number of bits for magnitude.
//     // Add 1 more bit for the sign if needed by the representation,
//     // but Python's RangeEstimate seems to return the bits for magnitude.
//     // Let's stick to magnitude bits.
//     uint32_t u_max_abs_val = (uint32_t)max_abs_val;
//     while (u_max_abs_val > 0) {
//         u_max_abs_val >>= 1;
//         bits++;
//     }
//      // Example: max_abs=127 (0111 1111). bits becomes 7. Correct.
//      // Example: max_abs=128 (1000 0000). bits becomes 8. Correct.
//     return bits;

//     // Alternative using floating point (requires linking -lm):
//     // return (int)ceil(log2((double)max_abs_val + 1e-9)); // Add epsilon
// }

// Overload for Matrix32 (assuming Matrix32 structure exists)
// static inline int range_estimate_matrix32(const Matrix32* matrix) {
//     if (!matrix || !matrix->matrix || matrix->width == 0 || matrix->height == 0) return 0;
//     int32_t max_abs_val = 0;
//     for (size_t i = 0; i < matrix->width; ++i) {
//         for (size_t j = 0; j < matrix->height; ++j) {
//             int32_t abs_val = abs(matrix->matrix[i][j]);
//             if (abs_val > max_abs_val) {
//                 max_abs_val = abs_val;
//             }
//         }
//     }
//     if (max_abs_val == 0) return 0;
//     int bits = 0;
//     uint32_t u_max_abs_val = (uint32_t)max_abs_val;
//     while (u_max_abs_val > 0) {
//         u_max_abs_val >>= 1;
//         bits++;
//     }
//     return bits;
// }

