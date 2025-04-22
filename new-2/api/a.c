// #include "network.h"
// #include <stdint.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h> // For memcpy
// #include "../ext/float_ops.h"

// // Global or static variable for mu
// static int8_t global_mu = 4; // Default

// void set_mu(int8_t value) {
//     global_mu = value;
// }

// // --- Initialization --- (Mostly unchanged, ensure scale is initialized)
// Network* init_network(uint8_t num_layers, uint8_t batch_size) {
//     Network* net = malloc(sizeof(Network));
//     net->num_layers = num_layers;
//     net->batch_size = batch_size;
//     net->layers = malloc(num_layers * sizeof(Layer*));
//     // Initialize layer pointers to NULL or handle allocation
//     for(int i=0; i<num_layers; ++i) net->layers[i] = NULL;
//     return net;
// }

// Layer* init_layer(uint8_t batch_size, uint16_t num_inputs, uint16_t num_neurons, LayerType type, Layer* next_layer) {
//     Layer* l = malloc(sizeof(Layer));
//     l->type = type;
//     l->next_layer = next_layer; // Store if needed

//     // Initialize matrices with scale = 0 initially
//     if (type == LINEAR) {
//         l->weights = init_m8(num_neurons, num_inputs); // fan_out x fan_in
//         l->weights.scale = 0; // Will be set by init_weights_xavier_uniform
//     } else {
//         // ReLU etc. don't have weights
//         // Use dummy matrix? Or handle NULL weights. Let's use dummy for safety.
//         l->weights = init_m8(0,0); // Or handle NULL checks later
//     }
//     l->activations = init_m8(batch_size, num_neurons);
//     l->activations.scale = 0; // Will be set during forward pass
//     l->input_copy = init_m8(batch_size, num_inputs);
//     l->input_copy.scale = 0; // Will be set during forward pass

//     l->act_exp = 0; // Initialize exponents
//     l->input_exp = 0;

//     return l;
// }

// // --- Freeing Functions --- (Mostly unchanged)
// // ... free_layer, free_network ...

// // --- Activation/Error/Gradient Calculations (UPDATED) ---

// Matrix8 act_calc(Matrix32* int32_acc, int8_t input_exp) {
//     // input_exp = exponent of input + exponent of weight
//     int8_t int32_bitwidth = estimate_matrix_bitwidth(int32_acc);
//     int8_t shift = int32_bitwidth - BITWIDTH;
//     int8_t exp_out;
//     Matrix8 result = init_m8(int32_acc->width, int32_acc->height);

//     // --- DEBUG LOG ---
//     #if DEBUG_LOG_LEVEL >= 2 // Make this more verbose
//     printf("DEBUG [act_calc]: InExp=%d, BW=%d, Shift=%d", input_exp, int32_bitwidth, shift);
//     #endif
//     // --- END DEBUG ---

//     if (shift > 0) {
//         exp_out = input_exp + shift;
//         for (uint16_t i = 0; i < result.width; ++i) {
//             for (uint16_t j = 0; j < result.height; ++j) {
//                 // Use RoundShift for activations
//                 result.matrix[i][j] = round_shift(int32_acc->matrix[i][j], shift);
//             }
//         }
//     } else {
//         exp_out = input_exp; // No shift applied
//         for (uint16_t i = 0; i < result.width; ++i) {
//             for (uint16_t j = 0; j < result.height; ++j) {
//                 // Clamp directly if no shift needed
//                 int32_t val = int32_acc->matrix[i][j];
//                 if (val > 127) result.matrix[i][j] = 127;
//                 else if (val < -128) result.matrix[i][j] = -128;
//                 else result.matrix[i][j] = (int8_t)val;
//             }
//         }
//     }
//     result.scale = exp_out;

//     // --- DEBUG LOG ---
//     #if DEBUG_LOG_LEVEL >= 2
//     printf(", OutExp=%d, FirstVal=%d\n", exp_out, result.matrix[0][0]);
//     #endif
//     // --- END DEBUG ---

//     return result;
// }

// Matrix8 err_calc(Matrix32* int32_acc, int8_t input_exp) {
//     // input_exp = exponent of incoming error + exponent of weight
//     int8_t int32_bitwidth = estimate_matrix_bitwidth(int32_acc);
//     int8_t shift = int32_bitwidth - BITWIDTH;
//     int8_t exp_out;
//     Matrix8 result = init_m8(int32_acc->width, int32_acc->height);

//     // --- DEBUG LOG ---
//     #if DEBUG_LOG_LEVEL >= 2 // Make this more verbose
//     printf("DEBUG [err_calc]: InExp=%d, BW=%d, Shift=%d", input_exp, int32_bitwidth, shift);
//     #endif
//     // --- END DEBUG ---

//     if (shift > 0) {
//         // Python err_calc uses shift as exponent directly? Let's re-check ti_torch
//         // Python: exp_out = shift; temp = ERROR_ROUND_METHOD(int32_acc, shift)
//         // This seems different from act_calc. Let's follow Python's err_calc logic:
//         exp_out = shift; // The exponent represents the scaling applied *during* the shift
//         for (uint16_t i = 0; i < result.width; ++i) {
//             for (uint16_t j = 0; j < result.height; ++j) {
//                 // Use RoundShift for errors
//                 result.matrix[i][j] = round_shift(int32_acc->matrix[i][j], shift);
//             }
//         }
//     } else {
//         exp_out = 0; // No shift applied, exponent is 0 relative to the input
//         for (uint16_t i = 0; i < result.width; ++i) {
//             for (uint16_t j = 0; j < result.height; ++j) {
//                 int32_t val = int32_acc->matrix[i][j];
//                 if (val > 127) result.matrix[i][j] = 127;
//                 else if (val < -128) result.matrix[i][j] = -128;
//                 else result.matrix[i][j] = (int8_t)val;
//             }
//         }
//     }
//     result.scale = exp_out;

//     // --- DEBUG LOG ---
//     #if DEBUG_LOG_LEVEL >= 2
//     printf(", OutExp=%d, FirstVal=%d\n", exp_out, result.matrix[0][0]);
//     #endif
//     // --- END DEBUG ---

//     return result;
// }


// // Returns int8 gradient matrix, sets out_shift via pointer
// Matrix8 grad_calc(Matrix32* int32_acc, int8_t mu, int8_t* out_shift) {
//     int8_t int32_bitwidth = estimate_matrix_bitwidth(int32_acc);
//     int8_t shift = int32_bitwidth - mu;
//     Matrix8 result = init_m8(int32_acc->width, int32_acc->height);
//     result.scale = 0; // Gradient scale is handled by the returned shift

//     // --- DEBUG LOG ---
//     #if DEBUG_LOG_LEVEL >= 1
//     printf("DEBUG [grad_calc]: mu=%d, BW=%d, CalcShift=%d", mu, int32_bitwidth, shift);
//     #endif
//     // --- END DEBUG ---

//     if (int32_bitwidth == 0) {
//         *out_shift = 0; // No gradient, shift is 0
//         // Matrix already initialized to zeros by init_m8
//     } else if (shift < 1) {
//         *out_shift = 0; // No effective shift applied
//         for (uint16_t i = 0; i < result.width; ++i) {
//             for (uint16_t j = 0; j < result.height; ++j) {
//                 int32_t val = int32_acc->matrix[i][j];
//                 if (val > 127) result.matrix[i][j] = 127;
//                 else if (val < -128) result.matrix[i][j] = -128;
//                 else result.matrix[i][j] = (int8_t)val;
//             }
//         }
//     } else {
//         *out_shift = shift; // Store the calculated shift
//         for (uint16_t i = 0; i < result.width; ++i) {
//             for (uint16_t j = 0; j < result.height; ++j) {
//                 // Use PstoShift for gradients
//                 result.matrix[i][j] = psto_shift(int32_acc->matrix[i][j], shift);
//             }
//         }
//     }

//     // --- DEBUG LOG ---
//     #if DEBUG_LOG_LEVEL >= 1
//     printf(", FinalShift=%d, FirstGradVal=%d\n", *out_shift, result.matrix[0][0]);
//     #endif
//     // --- END DEBUG ---

//     return result;
// }

// // In network.c
// void weight_update(Layer* layer, Matrix8* grad_int8, int8_t grad_exp) {
//     // grad_exp is now ignored in this simplified update function,
//     // assuming grad_int8 is already appropriately scaled by grad_calc.

//     // --- DEBUG LOG ---
//     #if DEBUG_LOG_LEVEL >= 1
//     static int weight_update_log_count = 0;
//     int should_log = (weight_update_log_count < 10); // Log first 10 updates overall
//     int8_t current_weight_00 = 0; // Initialize for logging
//     int8_t gradient_val_00 = 0;   // Initialize for logging
//     if (should_log && layer->weights.width > 0 && layer->weights.height > 0 && grad_int8->width > 0 && grad_int8->height > 0) {
//          current_weight_00 = layer->weights.matrix[0][0];
//          gradient_val_00 = grad_int8->matrix[0][0];
//          printf("DEBUG [WeightUpdate-Simple]: G[0][0]=%d, W_Before[0][0]=%d\n",
//                 gradient_val_00, current_weight_00);
//     }
//     #endif
//     // --- END DEBUG ---

//     for (uint16_t i = 0; i < layer->weights.width; ++i) { // Num neurons
//         for (uint16_t j = 0; j < layer->weights.height; ++j) { // Num inputs
//             int8_t current_weight = layer->weights.matrix[i][j];
//             int8_t gradient_val = grad_int8->matrix[i][j];

//             // Direct subtraction using int16_t for intermediate result
//             int16_t updated_weight_int16 = (int16_t)current_weight - (int16_t)gradient_val;

//             // Clamp result back to int8 range
//             if (updated_weight_int16 > 127) {
//                 layer->weights.matrix[i][j] = 127;
//             } else if (updated_weight_int16 < -128) {
//                 layer->weights.matrix[i][j] = -128;
//             } else {
//                 layer->weights.matrix[i][j] = (int8_t)updated_weight_int16;
//             }

//              // --- DEBUG LOG ---
//             #if DEBUG_LOG_LEVEL >= 1
//             if (should_log && i == 0 && j == 0) { // Log only the first weight element's update
//                 printf("DEBUG [WeightUpdate-Simple]: W_After[0][0]=%d\n", layer->weights.matrix[0][0]);
//                 // Check if weight actually changed
//                 if (layer->weights.matrix[0][0] == current_weight_00 && gradient_val_00 != 0) {
//                      printf("DEBUG [WeightUpdate-Simple]: WARNING - Weight[0][0] did not change despite non-zero gradient!\n");
//                 }
//                  weight_update_log_count++; // Increment counter after logging one update
//             }
//             #endif
//             // --- END DEBUG ---

//         } // end loop j
//     } // end loop i
//     // Weight exponent (layer->weights.scale) remains unchanged
// }



// // --- Layer Forward/Backward (UPDATED) ---

// Matrix8 layer_forward(Layer* layer, Matrix8* input) {
//     // Store input (quantized int8 and its exponent)
//     // Use m_cpy? Ensure scale is copied.
//     free_m8(&layer->input_copy); // Free previous copy if necessary
//     layer->input_copy = m_cpy(input); // Assumes m_cpy copies scale
//     layer->input_exp = input->scale; // Store exponent explicitly

//     if (layer->type == LINEAR) {
//         // Matrix multiplication: input * weights^T
//         // Input: (batch, features_in), Scale: input->scale
//         // Weight:(neurons_out, features_in), Scale: layer->weights.scale
//         // Result:(batch, neurons_out), Scale: input->scale + layer->weights.scale (before act_calc)
//         Matrix32 temp = get_mul8_2t(&layer->input_copy, &layer->weights);

//         // Calculate activations and output exponent
//         Matrix8 result = act_calc(&temp, layer->input_exp + layer->weights.scale);
//         free_m32(&temp); // Free intermediate 32-bit result

//         // Store activation and its exponent
//         free_m8(&layer->activations); // Free previous
//         layer->activations = m_cpy(&result); // Store result
//         layer->act_exp = result.scale; // Store exponent

//         return result; // result contains data and scale

//     } else if (layer->type == RELU) {
//         // ReLU: max(input, 0)
//         // Exponent does not change through ReLU
//         Matrix8 result = init_m8(input->width, input->height);
//         result.scale = input->scale; // Copy input exponent
//         for (uint16_t i = 0; i < input->width; ++i) {
//             for (uint16_t j = 0; j < input->height; ++j) {
//                 result.matrix[i][j] = (input->matrix[i][j] > 0) ? input->matrix[i][j] : 0;
//             }
//         }
//         // Store activation and its exponent
//         free_m8(&layer->activations);
//         layer->activations = m_cpy(&result);
//         layer->act_exp = result.scale;

//         return result;
//     }
//     // Add other layer types (FLATTEN, etc.) if needed
//     else {
//         fprintf(stderr, "Error: Unsupported layer type in layer_forward.\n");
//         // Return a dummy matrix or handle error
//         return init_m8(0,0);
//     }
// }

// Matrix8 layer_backward(Layer* layer, Matrix8* error_in) {
//     // error_in contains the int8 error and its exponent (error_in->scale)
//     int8_t err_in_exp = error_in->scale;

//     if (layer->type == LINEAR) {
//         // Calculate error to propagate back: error_in * weights
//         // error_in: (batch, neurons_out), Scale: err_in_exp
//         // weights:  (neurons_out, neurons_in), Scale: layer->weights.scale
//         // err_out_int32: (batch, neurons_in), Scale: err_in_exp + layer->weights.scale (before err_calc)
//         Matrix32 err_out_int32 = get_mul8(error_in, &layer->weights);

//         // Calculate int8 error_out and its exponent
//         // Pass the combined exponent before err_calc
//         Matrix8 err_out = err_calc(&err_out_int32, err_in_exp + layer->weights.scale);
//         free_m32(&err_out_int32);

//         // Calculate gradient: error_in^T * input_copy
//         // error_in^T: (neurons_out, batch)
//         // input_copy: (batch, neurons_in), Scale: layer->input_exp
//         // grad_int32acc: (neurons_out, neurons_in), Scale: err_in_exp + layer->input_exp (before grad_calc)
//         Matrix32 grad_int32acc = get_mul8_1t(error_in, &layer->input_copy);

//         // Calculate int8 gradient and the shift applied
//         int8_t grad_shift = 0;
//         Matrix8 grad_int8 = grad_calc(&grad_int32acc, global_mu, &grad_shift);
//         free_m32(&grad_int32acc);

//         // Calculate the final gradient exponent
//         int8_t grad_exp = err_in_exp + grad_shift + layer->input_exp;

//         // --- DEBUG LOG ---
//         #if DEBUG_LOG_LEVEL >= 1
//         static int layer_bwd_log_count = 0;
//          // Log less frequently, e.g., first few times or specific layer
//         if (layer_bwd_log_count < 10 /* && layer == specific_layer_pointer_if_needed */) {
//              printf("DEBUG [LayerBwd]: ErrInExp=%d, GradShift=%d, InputExp=%d => GradExp=%d\n",
//                     err_in_exp, grad_shift, layer->input_exp, grad_exp);
//              layer_bwd_log_count++;
//         }
//         #endif
//         // --- END DEBUG ---

//         // Update weights
//         weight_update(layer, &grad_int8, grad_exp);
//         free_m8(&grad_int8); // Free the int8 gradient matrix

//         return err_out; // Contains data and scale

//     } else if (layer->type == RELU) {
//         // Backprop through ReLU: error_out = error_in if activation > 0 else 0
//         Matrix8 err_out = init_m8(error_in->width, error_in->height);
//         err_out.scale = err_in_exp; // Exponent doesn't change

//         for (uint16_t i = 0; i < err_out.width; ++i) {
//             for (uint16_t j = 0; j < err_out.height; ++j) {
//                 // Use the stored activation from the forward pass
//                 err_out.matrix[i][j] = (layer->activations.matrix[i][j] > 0) ? error_in->matrix[i][j] : 0;
//             }
//         }
//         return err_out;
//     }
//     // Add other layer types
//     else {
//          fprintf(stderr, "Error: Unsupported layer type in layer_backward.\n");
//          return init_m8(0,0);
//     }
// }

// // --- Network Forward/Backward (UPDATED) ---

// Matrix8 network_forward(Network* network, Matrix8* X) {
//     Matrix8 current_activations = m_cpy(X); // Start with input, copy scale

//     for (uint8_t i = 0; i < network->num_layers; ++i) {
//         Matrix8 next_activations = layer_forward(network->layers[i], &current_activations);
//         free_m8(&current_activations); // Free intermediate activations
//         current_activations = next_activations; // Keep the latest (contains data and scale)
//     }
//     // The final current_activations is the output of the last layer
//     return current_activations;
// }

// Matrix8 network_backward(Network* network, Vector8* Y) {
//     // Y contains target labels (int8_t), assumed scale 0

//     // 1. Get final layer's activation (already computed in forward pass)
//     Layer* last_layer = network->layers[network->num_layers - 1];
//     Matrix8 out_activations = last_layer->activations; // This has scale last_layer->act_exp

//     // 2. Calculate initial error gradient (Float calculation -> Quantize)
//     uint16_t batch_size = out_activations.width;
//     uint16_t num_classes = out_activations.height;

//     // Allocate temporary float storage
//     float** float_activations = malloc(batch_size * sizeof(float*));
//     float** float_error = malloc(batch_size * sizeof(float*));
//     for(uint16_t i=0; i<batch_size; ++i) {
//         float_activations[i] = malloc(num_classes * sizeof(float));
//         float_error[i] = malloc(num_classes * sizeof(float));
//     }

//     // Reconstruct float activations
//     float act_scale_factor = powf(2.0f, (float)out_activations.scale);
//     for(uint16_t i=0; i<batch_size; ++i) {
//         for(uint16_t j=0; j<num_classes; ++j) {
//             float_activations[i][j] = (float)out_activations.matrix[i][j] * act_scale_factor;
//         }
//     }

//     // Calculate float error (activation - target)
//     // Assuming cross-entropy gradient for softmax is approximated by (pred - target)
//     // Or for simple MSE-like loss on logits: (pred - target)
//     for(uint16_t i=0; i<batch_size; ++i) {
//         int8_t target_label = Y->vector[i];
//         for(uint16_t j=0; j<num_classes; ++j) {
//             float target_val = (j == target_label) ? 1.0f : 0.0f; // One-hot target
//             // Adjust target scaling if necessary? Assume target is ideal 0/1 for now.
//             // Error = Prediction - Target
//             float_error[i][j] = float_activations[i][j] - target_val;
//             // If using softmax/cross-entropy, the gradient is simpler:
//             // float_error[i][j] = softmax_output[i][j] - target_val;
//             // But we only have logits (activations). Using (logit - target) is a common simplification.
//         }
//     }

//     // Quantize float error using adaptive method
//     Matrix8 current_error = quantize_float_matrix_adaptive(float_error, batch_size, num_classes);

//     // --- DEBUG LOG ---
//     #if DEBUG_LOG_LEVEL >= 1
//     static int initial_err_log_count = 0;
//     if (initial_err_log_count < 2) { // Log first couple of batches
//         printf("DEBUG [Initial Error]: LastActExp=%d, ErrExp=%d, Err[0][0]=%d\n",
//                last_layer->act_exp, current_error.scale, current_error.matrix[0][0]);
//         initial_err_log_count++;
//     }
//     #endif

//     // Free temporary float storage
//     for(uint16_t i=0; i<batch_size; ++i) {
//         free(float_activations[i]);
//         free(float_error[i]);
//     }
//     free(float_activations);
//     free(float_error);

//     // 3. Propagate error back through layers
//     for (int i = network->num_layers - 1; i >= 0; --i) {
//         Matrix8 prev_error = layer_backward(network->layers[i], &current_error);
//         free_m8(&current_error);
//         current_error = prev_error;
//     }

//     // current_error now holds the error propagated back to the input
//     return current_error;
// }

// Matrix8 m_cpy_range(Matrix8* orig, uint8_t start, uint8_t end) {
//     Matrix8 result = init_m8(min(end, orig->width) - start + 1, orig->height);
//     for (uint8_t i = 0; i < result.width; ++i) {
//         for (uint8_t j = 0; j < result.height; ++j) {
//             result.matrix[i][j] = orig->matrix[i + start][j];
//         }
//     }
//     result.scale = orig->scale;
//     return result;
// }

// Vector8 v_cpy_range(Vector8* orig, uint8_t start, uint8_t end) {
//     Vector8 result;
//     result.length = end - start + 1;
//     result.vector = malloc(result.length * sizeof(int8_t));
//     result.scale = orig->scale;
//     for (uint8_t i = 0; i < result.length; ++i) {
//         result.vector[i] = orig->vector[i + start];
//     }
//     return result;
// }

// Matrix8 m_cpy(Matrix8 *m) {
//     Matrix8 result = init_m8(m->width, m->height);
//     for (uint16_t i = 0; i < m->width; ++i) {
//         for (uint16_t j = 0; j < m->height; ++j) {
//             result.matrix[i][j] = m->matrix[i][j];
//         }
//     }
//     result.scale = m->scale;
//     return result;
// }

// void free_network(Network* network) {
//     for (uint8_t i = 0; i < network->num_layers; ++i) {
//         free_layer(network->layers[i]);
//     }
//     free(network->layers);
//     free(network);
// }

// void free_layer(Layer* layer) {
//     free_m8(&layer->weights);
//     free_m8(&layer->activations);
//     free(layer);
// }

// void print_layer(Layer* layer, char* name) {
//     printf("[%s] Layer:\n", name);
//     print_matrix8(&layer->weights, "Weights");
//     print_matrix8(&layer->activations, "Activations");
// };