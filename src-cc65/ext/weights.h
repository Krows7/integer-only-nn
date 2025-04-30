#ifndef WEIGHTS_H
#define WEIGHTS_H

#include "linear.h"

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
void init_weights_xavier_uniform(Matrix8* weights);

#endif // WEIGHTS_H
