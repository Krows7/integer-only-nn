#include <math.h>     // For sqrtf, log2f, ceilf, fabsf, roundf, powf
#include <stdlib.h>   // For rand, malloc, free
#include <stdint.h>   // For int8_t
#include <float.h>    // For FLT_MIN or DBL_MIN (optional, can use small epsilon)
#include <stdio.h>    // For error printing (optional)
#include "linear-math.h" // Assuming Matrix8 definition is here

// Define the target bitwidth for weight quantization (matches Python BITWIDTH=7)
// This means the effective range is roughly -127 to +127 after scaling.
#define WEIGHT_QUANT_BITWIDTH 7

// Helper function to clamp a float before casting to int8_t
// Ensures the value is within [-128, 127]
static inline int8_t clamp_to_int8(float val) {
    // Round to nearest integer first
    val = roundf(val);
    // Clamp to the valid range of int8_t
    if (val > 127.0f) {
        return 127;
    } else if (val < -128.0f) {
        return -128;
    } else {
        return (int8_t)val;
    }
}

/**
 * @brief Initializes a Matrix8 with weights using Xavier Uniform initialization
 *        and quantizes them to int8_t, storing the quantization scale (exponent).
 *
 * @param weights Pointer to a pre-allocated Matrix8 structure.
 *                The function assumes weights->matrix is allocated with dimensions
 *                weights->width (num_neurons/fan_out) x weights->height (num_inputs/fan_in).
 *                The function will fill weights->matrix and set weights->scale.
 * @note Requires srand() to be called once elsewhere (e.g., in main)
 *       before using this function.
 */
void init_weights_xavier_uniform(Matrix8* weights) {
    if (!weights || !weights->matrix || weights->width == 0 || weights->height == 0) {
        fprintf(stderr, "Error: Invalid or unallocated Matrix8 provided to init_weights_xavier_uniform.\n");
        // Consider exiting or asserting depending on error handling strategy
        return;
    }

    // fan_out = number of output neurons (width of the matrix)
    // fan_in = number of input features (height of the matrix)
    int fan_out = weights->width;
    int fan_in = weights->height;
    size_t num_elements = (size_t)fan_out * fan_in;

    // --- 1. Calculate Xavier Uniform bounds 'a' ---
    // a = gain * sqrt(6 / (fan_in + fan_out))
    // Using gain = 1.0f as is common for linear/sigmoid/tanh
    float a = sqrtf(6.0f / (float)(fan_in + fan_out));

    // --- 2. Generate temporary float weights and find max absolute value ---
    float* temp_weights = (float*)malloc(num_elements * sizeof(float));
    if (!temp_weights) {
        perror("Error allocating temporary float weights");
        // Consider exiting or asserting
        return;
    }

    float max_abs_val = 0.0f;
    for (size_t i = 0; i < num_elements; ++i) {
        // Generate random float in [0, 1]
        float rand_f = (float)rand() / (float)RAND_MAX;
        // Scale and shift to [-a, a]
        temp_weights[i] = (rand_f * 2.0f * a) - a;

        // Track maximum absolute value for quantization range
        float abs_val = fabsf(temp_weights[i]);
        if (abs_val > max_abs_val) {
            max_abs_val = abs_val;
        }
    }

    // --- 3 & 4. Calculate quantization parameters (scale/exponent) ---
    int8_t exponent = 0; // Default exponent if all weights are zero
    float scale_factor = 1.0f; // Default scale factor

    // Use a small epsilon to avoid issues with log2f(0) or near-zero values
    if (max_abs_val > 1e-9f) {
        // Calculate the bitwidth needed to represent the float range
        float input_bitwidth = ceilf(log2f(max_abs_val));

        // Calculate the exponent needed to shift the range into WEIGHT_QUANT_BITWIDTH bits
        // exponent = input_bitwidth - target_bitwidth
        exponent = (int8_t)(input_bitwidth - (float)WEIGHT_QUANT_BITWIDTH);

        // Calculate the scaling factor to apply before rounding
        // scale_factor = 2^(target_bitwidth - input_bitwidth) = 2^(-exponent)
        scale_factor = powf(2.0f, (float)WEIGHT_QUANT_BITWIDTH - input_bitwidth);
    }

    // --- 5 & 6. Quantize floats to int8_t and store in Matrix8 ---
    for (int w = 0; w < fan_out; ++w) { // Iterate through rows (neurons)
        for (int h = 0; h < fan_in; ++h) { // Iterate through columns (inputs)
            // Calculate linear index assuming row-major storage internally if needed,
            // but direct access weights->matrix[w][h] is typical for Matrix8.
            size_t idx = (size_t)w * fan_in + h; // Index into temp_weights

            // Apply scaling factor
            float normalized_val = temp_weights[idx] * scale_factor;

            // Round, clamp, and store the int8_t value
            weights->matrix[w][h] = clamp_to_int8(normalized_val);
        }
    }

    // Store the calculated exponent in the matrix's scale field
    weights->scale = exponent;

    // --- Cleanup ---
    free(temp_weights);
}