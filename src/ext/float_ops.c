#include "float_ops.h"
#include "quantization.h"

// --- Helper: Clamp float to int8 range ---
int8_t clamp_to_int8(float val) {
    val = roundf(val);
    if (val > 127.0f) return 127;
    if (val < -128.0f) return -128;
    return (int8_t)val;
}

// --- Adaptive Quantization (Float -> Int8) ---
Matrix8 quantize_float_matrix_adaptive(float** float_matrix, lsize_t width, lsize_t height) {
    Matrix8 result = init_m8(width, height); // Assumes init_m8 allocates memory
    float max_abs_val = 0.0f;

    // 1. Find max absolute value
    for (lsize_t i = 0; i < width; ++i) {
        for (lsize_t j = 0; j < height; ++j) {
            float abs_val = fabsf(float_matrix[i][j]);
            if (abs_val > max_abs_val) {
                max_abs_val = abs_val;
            }
        }
    }

    // 2. Calculate exponent
    int8_t exponent = 0;
    float scale_factor = 1.0f;
    if (max_abs_val > 1e-9f) { // Avoid log(0)
        float input_bitwidth = ceilf(log2f(max_abs_val));
        exponent = (int8_t)(input_bitwidth - (float)BITWIDTH);
        // scale_factor = 2^(BITWIDTH - input_bitwidth) = 2^(-exponent)
        scale_factor = powf(2.0f, (float)BITWIDTH - input_bitwidth);
    }

    // 3. Scale, round, clamp, and store
    for (lsize_t i = 0; i < width; ++i) {
        for (lsize_t j = 0; j < height; ++j) {
            float normalized_val = float_matrix[i][j] * scale_factor;
            result.matrix[i][j] = clamp_to_int8(normalized_val);
        }
    }
    result.scale = exponent;
    return result;
}

// Vector version (similar logic)
Vector8 quantize_float_vector_adaptive(const float* float_vector, lsize_t length) {
    Vector8 result = init_v8(length);
    float max_abs_val = 0.0f;
    for (lsize_t i = 0; i < length; ++i) {
        float abs_val = fabsf(float_vector[i]);
        if (abs_val > max_abs_val) max_abs_val = abs_val;
    }

    int8_t exponent = 0;
    float scale_factor = 1.0f;
    if (max_abs_val > 1e-9f) {
        float input_bitwidth = ceilf(log2f(max_abs_val));
        exponent = (int8_t)(input_bitwidth - (float)BITWIDTH);
        scale_factor = powf(2.0f, (float)BITWIDTH - input_bitwidth);
    }

    for (lsize_t i = 0; i < length; ++i) {
        float normalized_val = float_vector[i] * scale_factor;
        result.vector[i] = clamp_to_int8(normalized_val);
    }
    result.scale = exponent;
    return result;
}