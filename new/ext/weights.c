#include "weights.h"

// Ensure BITWIDTH is used here, matching the rest of the code
// #define WEIGHT_QUANT_BITWIDTH 7 // Defined in header now

// ... clamp_to_int8 if not already included ...

void init_weights_xavier_uniform(Matrix8* weights) {
    if (!weights || !weights->matrix || weights->width == 0 || weights->height == 0) {
        fprintf(stderr, "Error: Invalid Matrix8 in init_weights_xavier_uniform.\n");
        return;
    }

    int fan_out = weights->width;
    int fan_in = weights->height;
    size_t num_elements = (size_t)fan_out * fan_in;

    float a = sqrtf(6.0f / (float)(fan_in + fan_out));

    float* temp_weights = (float*)malloc(num_elements * sizeof(float));
    if (!temp_weights) { /* error handling */ return; }

    float max_abs_val = 0.0f;
    for (size_t i = 0; i < num_elements; ++i) {
        float rand_f = (float)rand() / (float)RAND_MAX;
        temp_weights[i] = (rand_f * 2.0f * a) - a;
        float abs_val = fabsf(temp_weights[i]);
        if (abs_val > max_abs_val) max_abs_val = abs_val;
    }

    int8_t exponent = 0;
    float scale_factor = 1.0f;

    if (max_abs_val > 1e-9f) {
        float input_bitwidth = ceilf(log2f(max_abs_val));
        // Use BITWIDTH from header
        exponent = (int8_t)(input_bitwidth - (float)BITWIDTH);
        scale_factor = powf(2.0f, (float)BITWIDTH - input_bitwidth);
    }

    for (int w = 0; w < fan_out; ++w) {
        for (int h = 0; h < fan_in; ++h) {
            size_t idx = (size_t)w * fan_in + h;
            float normalized_val = temp_weights[idx] * scale_factor;
            weights->matrix[w][h] = clamp_to_int8(normalized_val);
        }
    }

    weights->scale = exponent; // Store the exponent

    free(temp_weights);
}