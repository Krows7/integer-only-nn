#ifndef FLOAT_OPS_H
#define FLOAT_OPS_H

#include "linear.h"

// --- New Quantization Functions ---

// Adaptive quantization for float matrices (TiFloatToInt8 equivalent)
Matrix8 quantize_float_matrix_adaptive(float** float_matrix, lsize_t width, lsize_t height);
// Helper for single vector (useful for initial error)
Vector8 quantize_float_vector_adaptive(const float* float_vector, lsize_t length);

int8_t clamp_to_int8(float val);

#endif