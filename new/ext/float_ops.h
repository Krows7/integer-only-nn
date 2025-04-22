#ifndef QUANTIZATION_H
#define QUANTIZATION_H

#include "../api/quantization.h"

// --- New Quantization Functions ---

// Adaptive quantization for float matrices (TiFloatToInt8 equivalent)
Matrix8_ext quantize_float_matrix_adaptive(float** float_matrix, uint16_t width, uint16_t height);
// Helper for single vector (useful for initial error)
Vector8 quantize_float_vector_adaptive(float* float_vector, uint16_t length);

int8_t clamp_to_int8(float val);

#endif