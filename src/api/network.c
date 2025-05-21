#include "network.h"
#include "linear.h"
#include "quantization.h"
#include <inttypes.h>
#include <stdint.h>

static int8_t global_mu = 4;

void set_mu(int8_t value) {
    global_mu = value;
}

__bank(1) Network* init_network(lsize_t num_layers, lsize_t batch_size) {
    Network* net = malloc(sizeof(Network));
    net->num_layers = num_layers;
    net->layers = malloc(num_layers * sizeof(Layer*));
    net->batch_size = batch_size;
    for(lsize_t i=0; i < num_layers; ++i) net->layers[i] = NULL;
    return net;
}

__bank(1) Layer* init_layer(lsize_t batch_size, lsize_t num_inputs, lsize_t num_neurons, LayerType type) {
    Layer* l = malloc(sizeof(Layer));
    l->type = type;
    if (type == LINEAR) {
        l->weights = init_m8(num_neurons, num_inputs); // fan_out x fan_in
    }
    // Initial activations don't need before training
    // l->activations = init_m8(batch_size, num_neurons);
    l->activations = init_m8(0, 0);
    l->input_copy = init_m8(0, 0);
    // println("%d %d %d %d %d %d", l->weights.width, l->weights.height, l->activations.width, l->activations.height, l->input_copy.width, l->input_copy.height);
    return l;
}

__bank(2) Matrix8 act_calc(Matrix32* int32_acc, int8_t input_exp) {
    // input_exp = exponent of input + exponent of weight
    int8_t int32_bitwidth = estimate_matrix_bitwidth(int32_acc);
    int8_t shift = int32_bitwidth - BITWIDTH;
    int8_t exp_out;
    Matrix8 result = init_m8(int32_acc->width, int32_acc->height);
    // printf("Bitshift: %d\n", shift);

    debug("[act_calc]: InExp=%d, BW=%d, Shift=%d", input_exp, int32_bitwidth, shift);

    if (shift > 0) {
        exp_out = input_exp + shift;
        for (lsize_t i = 0; i < result.width; ++i) {
            // printf("%d %d\n\n", i, result.width);
            int8_t* resultI = result.matrix[i];
            const int32_t* inputI = int32_acc->matrix[i];
            for (lsize_t j = 0; j < result.height; ++j) {
                // Use RoundShift for activations
                // printf("%d %d %d %d\n", i, result.width, j, result.height);
                resultI[j] = round_shift(inputI[j], shift);
                // printf("%d %d %d %d\n\n", i, result.width, j, result.height);
            }
        }
    } else {
        // exp_out = input_exp; // No shift applied
        exp_out = input_exp + shift;
        for (lsize_t i = 0; i < result.width; ++i) {
            for (lsize_t j = 0; j < result.height; ++j) {
                // Clamp directly if no shift needed
                int32_t val = int32_acc->matrix[i][j] >> -shift;
                if (val > 127) result.matrix[i][j] = 127;
                else if (val < -128) result.matrix[i][j] = -128;
                else result.matrix[i][j] = (int8_t)val;
            }
        }
    }
    result.scale = exp_out;

    debug(", OutExp=%d, FirstVal=%d\n", exp_out, result.matrix[0][0]);

    return result;
}

void err_calc_1(Matrix32* int32_acc, Matrix8* out) {
    // input_exp = exponent of incoming error + exponent of weight
    int8_t int32_bitwidth = estimate_matrix_bitwidth(int32_acc);
    int8_t shift = int32_bitwidth - BITWIDTH;
    int8_t exp_out;
    if (out->width != int32_acc->width || out->height != int32_acc->height) {
        free_m8(out);
        *out = init_m8(int32_acc->width, int32_acc->height);
    }
    // Matrix8 result = init_m8(int32_acc->width, int32_acc->height);

    debug("[err_calc]: InExp=%d, BW=%d, Shift=%d", input_exp, int32_bitwidth, shift);

    if (shift > 0) {
        // Python err_calc uses shift as exponent directly? Let's re-check ti_torch
        // Python: exp_out = shift; temp = ERROR_ROUND_METHOD(int32_acc, shift)
        // This seems different from act_calc. Let's follow Python's err_calc logic:
        exp_out = shift; // The exponent represents the scaling applied *during* the shift
        for (lsize_t i = 0; i < out->width; ++i) {
            for (lsize_t j = 0; j < out->height; ++j) {
                // Use RoundShift for errors
                out->matrix[i][j] = round_shift(int32_acc->matrix[i][j], shift);
            }
        }
    } else {
        exp_out = 0; // No shift applied, exponent is 0 relative to the input
        for (lsize_t i = 0; i < out->width; ++i) {
            for (lsize_t j = 0; j < out->height; ++j) {
                int32_t val = int32_acc->matrix[i][j];
                if (val > 127) out->matrix[i][j] = 127;
                else if (val < -128) out->matrix[i][j] = -128;
                else out->matrix[i][j] = (int8_t)val;
            }
        }
    }
    out->scale = exp_out;

    debug(", OutExp=%d, FirstVal=%d\n", exp_out, result.matrix[0][0]);
}

__bank(2) Matrix8 err_calc(Matrix32* int32_acc) {
    // input_exp = exponent of incoming error + exponent of weight
    int8_t int32_bitwidth = estimate_matrix_bitwidth(int32_acc);
    int8_t shift = int32_bitwidth - BITWIDTH;
    int8_t exp_out;
    Matrix8 result = init_m8(int32_acc->width, int32_acc->height);

    debug("[err_calc]: InExp=%d, BW=%d, Shift=%d", input_exp, int32_bitwidth, shift);

    if (shift > 0) {
        // Python err_calc uses shift as exponent directly? Let's re-check ti_torch
        // Python: exp_out = shift; temp = ERROR_ROUND_METHOD(int32_acc, shift)
        // This seems different from act_calc. Let's follow Python's err_calc logic:
        exp_out = shift; // The exponent represents the scaling applied *during* the shift
        for (lsize_t i = 0; i < result.width; ++i) {
            for (lsize_t j = 0; j < result.height; ++j) {
                // Use RoundShift for errors
                result.matrix[i][j] = round_shift(int32_acc->matrix[i][j], shift);
            }
        }
    } else {
        exp_out = 0; // No shift applied, exponent is 0 relative to the input
        for (lsize_t i = 0; i < result.width; ++i) {
            for (lsize_t j = 0; j < result.height; ++j) {
                int32_t val = int32_acc->matrix[i][j];
                if (val > 127) result.matrix[i][j] = 127;
                else if (val < -128) result.matrix[i][j] = -128;
                else result.matrix[i][j] = (int8_t)val;
            }
        }
    }
    result.scale = exp_out;

    debug(", OutExp=%d, FirstVal=%d\n", exp_out, result.matrix[0][0]);

    return result;
}


// Returns int8 gradient matrix, sets out_shift via pointer
__bank(2) Matrix8 grad_calc(Matrix32* int32_acc, int8_t mu, int8_t* out_shift) {
    int8_t int32_bitwidth = estimate_matrix_bitwidth(int32_acc);
    int8_t shift = int32_bitwidth - mu;
    Matrix8 result = init_m8(int32_acc->width, int32_acc->height);

    debug("[grad_calc]: mu=%d, BW=%d, CalcShift=%d", mu, int32_bitwidth, shift);

    #ifdef LIN_DEBUG
    println("grad_calc shift: %d", shift);
    #endif

    if (int32_bitwidth == 0) {
        *out_shift = 0; // No gradient, shift is 0
        // Matrix already initialized to zeros by init_m8
    } else if (shift < 1) {
        *out_shift = 0; // No effective shift applied
        for (lsize_t i = 0; i < result.width; ++i) {
            for (lsize_t j = 0; j < result.height; ++j) {
                int32_t val = int32_acc->matrix[i][j];
                if (val > 127) result.matrix[i][j] = 127;
                else if (val < -128) result.matrix[i][j] = -128;
                else result.matrix[i][j] = (int8_t)val;
            }
        }
    } else {
        *out_shift = shift; // Store the calculated shift
        for (lsize_t i = 0; i < result.width; ++i) {
            for (lsize_t j = 0; j < result.height; ++j) {
                // Use PstoShift for gradients
                result.matrix[i][j] = psto_shift(int32_acc->matrix[i][j], shift);
            }
        }
        print_matrix8_d(&result, "grad_calc Result");
    }

    // print_matrix32(int32_acc, "int32_acc");
    // print_matrix8(&result, "grad_calc");

    debug(", FinalShift=%d, FirstGradVal=%d\n", *out_shift, result.matrix[0][0]);

    return result;
}

__bank(2) void weight_update(const Layer* layer, const Matrix8* grad_int8) {
    // grad_exp is now ignored in this simplified update function,
    // assuming grad_int8 is already appropriately scaled by grad_calc.

    // --- DEBUG LOG ---
    #if DEBUG_LOG_LEVEL >= 2
    static int weight_update_log_count = 0;
    int should_log = (weight_update_log_count < 10); // Log first 10 updates overall
    int8_t current_weight_00 = 0; // Initialize for logging
    int8_t gradient_val_00 = 0;   // Initialize for logging
    if (should_log && layer->weights.width > 0 && layer->weights.height > 0 && grad_int8->width > 0 && grad_int8->height > 0) {
         current_weight_00 = layer->weights.matrix[0][0];
         gradient_val_00 = grad_int8->matrix[0][0];
         printf("DEBUG [WeightUpdate-Simple]: G[0][0]=%d, W_Before[0][0]=%d\n",
                gradient_val_00, current_weight_00);
    }
    #endif
    // --- END DEBUG ---

    for (lsize_t i = 0; i < layer->weights.width; ++i) { // Num neurons
        for (lsize_t j = 0; j < layer->weights.height; ++j) { // Num inputs
            int8_t current_weight = layer->weights.matrix[i][j];
            int8_t gradient_val = grad_int8->matrix[i][j];

            // Direct subtraction using int16_t for intermediate result
            int16_t updated_weight_int16 = (int16_t)current_weight - (int16_t)gradient_val;

            // Clamp result back to int8 range
            if (updated_weight_int16 > 127) {
                layer->weights.matrix[i][j] = 127;
            } else if (updated_weight_int16 < -128) {
                layer->weights.matrix[i][j] = -128;
            } else {
                layer->weights.matrix[i][j] = (int8_t)updated_weight_int16;
            }

            #if DEBUG_LOG_LEVEL >= 2
            if (should_log && i == 0 && j == 0) { // Log only the first weight element's update
                printf("DEBUG [WeightUpdate-Simple]: W_After[0][0]=%d\n", layer->weights.matrix[0][0]);
                if (layer->weights.matrix[0][0] == current_weight_00 && gradient_val_00 != 0) {
                    printf("DEBUG [WeightUpdate-Simple]: WARNING - Weight[0][0] did not change despite non-zero gradient!\n");
                }
                weight_update_log_count++; // Increment counter after logging one update
            }
            #endif

        }
    }
    // Weight exponent (layer->weights.scale) remains unchanged
}



__bank(2) Matrix8 layer_forward(Layer* layer, const Matrix8* input) {
    // free_m8(&layer->input_copy); // Free previous copy if necessary
    // layer->input_copy = m_cpy(input); // Assumes m_cpy copies scale
    m_cpy(&layer->input_copy, input);
    if (layer->type == LINEAR) {
        Matrix32 temp = get_mul8_2t(&layer->input_copy, &layer->weights);

        // Calculate activations and output exponent
        Matrix8 result = act_calc(&temp, layer->input_copy.scale + layer->weights.scale);
        free_m32(&temp); // Free intermediate 32-bit result

        // Store activation and its exponent
        // free_m8(&layer->activations); // Free previous
        // layer->activations = m_cpy(&result); // Store result
        m_cpy(&layer->activations, &result);

        return result; // result contains data and scale
    } else if (layer->type == RELU) {
        // ReLU: max(input, 0)
        // Exponent does not change through ReLU
        Matrix8 result = init_m8(input->width, input->height);
        result.scale = input->scale; // Copy input exponent
        for (lsize_t i = 0; i < input->width; ++i) {
            for (lsize_t j = 0; j < input->height; ++j) {
                result.matrix[i][j] = (input->matrix[i][j] > 0) ? input->matrix[i][j] : 0;
            }
        }
        // Store activation and its exponent
        // free_m8(&layer->activations);
        // layer->activations = m_cpy(&result);
        m_cpy(&layer->activations, &result);

        return result;
    } else {
        error("Error: Unsupported layer type in layer_forward.");
        return init_m8(0,0);
    }
}

// __bank(2) Matrix8 layer_backward(const Layer* layer, const Matrix8* error_in) {
//     if (layer->type == LINEAR) {
//         // Calculate error to propagate back: error_in * weights
//         // error_in: (batch, neurons_out), Scale: err_in_exp
//         // weights:  (neurons_out, neurons_in), Scale: layer->weights.scale
//         // err_out_int32: (batch, neurons_in), Scale: err_in_exp + layer->weights.scale (before err_calc)
//         Matrix32 err_out_int32 = get_mul8(error_in, &layer->weights);

//         // Calculate gradient: error_in^T * input_copy
//         // error_in^T: (neurons_out, batch)
//         // input_copy: (batch, neurons_in), Scale: layer->input_exp
//         // grad_int32acc: (neurons_out, neurons_in), Scale: err_in_exp + layer->input_exp (before grad_calc)
//         Matrix32 grad_int32acc = get_mul8_1t(error_in, &layer->input_copy);

//         // Calculate int8 error_out and its exponent
//         // Pass the combined exponent before err_calc
//         Matrix8 err_out = err_calc(&err_out_int32);
//         free_m32(&err_out_int32);


//         print_matrix8_d(error_in, "ErrIn");
//         print_matrix8_d(&layer->input_copy, "InputCopy");
//         print_matrix32_d(&grad_int32acc, "grad_int32acc");

//         // Calculate int8 gradient and the shift applied
//         int8_t grad_shift = 0;
//         Matrix8 grad_int8 = grad_calc(&grad_int32acc, global_mu, &grad_shift);
//         free_m32(&grad_int32acc);

//         #if DEBUG_LOG_LEVEL >= 2
//         for (lsize_t i = 0; i < 10; ++i) {
//             debug("[LayerBwd]: ErrInExp=%d, GradShift=%d, InputExp=%d => GradExp=%d",
//                 err_in_exp, grad_shift, layer->input_copy.scale, grad_exp);
//         }
//         #endif

//         // Update weights
//         weight_update(layer, &grad_int8);
//         free_m8(&grad_int8); // Free the int8 gradient matrix

//         print_matrix8_d(&grad_int8, "Layer Backward Gradient");
//         print_matrix8_d(&err_out, "Layer Backward Linear");

//         return err_out; // Contains data and scale
//     } else if (layer->type == RELU) {
//         // Backprop through ReLU: error_out = error_in if activation > 0 else 0
//         Matrix8 err_out = init_m8(error_in->width, error_in->height);
//         err_out.scale = error_in->scale; // Exponent doesn't change

//         for (lsize_t i = 0; i < err_out.width; ++i) {
//             for (lsize_t j = 0; j < err_out.height; ++j) {
//                 // Use the stored activation from the forward pass
//                 err_out.matrix[i][j] = (layer->activations.matrix[i][j] > 0) ? error_in->matrix[i][j] : 0;
//             }
//         }

//         print_matrix8_d(&err_out, "Layer Backward ReLU");

//         return err_out;
//     }
//     error("Unsupported layer type in layer_backward.");
//     return init_m8(0,0);
// }

__bank(2) void layer_backward_1(const Layer* layer, const Matrix8* error_in, Matrix8* out) {
    if (layer->type == LINEAR) {
        Matrix32 err_out_int32 = get_mul8(error_in, &layer->weights);

        Matrix32 grad_int32acc = get_mul8_1t(error_in, &layer->input_copy);

        // Matrix8 err_out = err_calc(&err_out_int32);
        // err_calc_1(&err_out_int32, out);
        free_m8(out);
        *out = err_calc(&err_out_int32);
        free_m32(&err_out_int32);

        print_matrix8_d(error_in, "ErrIn");
        print_matrix8_d(&layer->input_copy, "InputCopy");
        print_matrix32_d(&grad_int32acc, "grad_int32acc");

        int8_t grad_shift = 0;
        Matrix8 grad_int8 = grad_calc(&grad_int32acc, global_mu, &grad_shift);
        free_m32(&grad_int32acc);

        #if DEBUG_LOG_LEVEL >= 2
        for (lsize_t i = 0; i < 10; ++i) {
            debug("[LayerBwd]: ErrInExp=%d, GradShift=%d, InputExp=%d => GradExp=%d",
                err_in_exp, grad_shift, layer->input_copy.scale, grad_exp);
        }
        #endif

        // Update weights
        weight_update(layer, &grad_int8);
        free_m8(&grad_int8); // Free the int8 gradient matrix

        print_matrix8_d(&grad_int8, "Layer Backward Gradient");
        print_matrix8_d(&err_out, "Layer Backward Linear");
    } else if (layer->type == RELU) {
        for (lsize_t i = 0; i < out->width; ++i) {
            for (lsize_t j = 0; j < out->height; ++j) {
                // Use the stored activation from the forward pass
                out->matrix[i][j] = (layer->activations.matrix[i][j] > 0) ? error_in->matrix[i][j] : 0;
            }
        }
        print_matrix8_d(&err_out, "Layer Backward ReLU");
    } else error("Unsupported layer type in layer_backward.");
}

__bank(2) void layer_forward_1(Layer* layer, const Matrix8* input, Matrix8* out) {
    m_cpy(&layer->input_copy, input);
    // printf("kek\n");
    if (layer->type == LINEAR) {
        // printf("kek1\n");
        Matrix32 temp = get_mul8_2t(&layer->input_copy, &layer->weights);
        print_matrix8_d(&layer->input_copy, "A");
        print_matrix8_d(&layer->weights, "B");
        print_matrix32_d(&temp, "layer_forward_1_temp");
        // printf("kek2\n");
        for (lsize_t i = 0; i < out->width; ++i) {
            free(out->matrix[i]);
        }
        // printf("ke3\n");
        free(out->matrix);
        // log("%d + %d = %d", layer->input_copy.scale, layer->weights.scale, layer->input_copy.scale + layer->weights.scale);
        // printf("Gonna act_calc\n");
        *out = act_calc(&temp, layer->input_copy.scale + layer->weights.scale);
        // printf("Gonna Free\n");
        free_m32(&temp);
        // printf("Gonna copy\n");
        m_cpy(&layer->activations, out);
        // printf("Layer Complete\n");
        // print_layer(layer, "Linear");
    } else if (layer->type == RELU) {
        if (!out->matrix) *out = init_m8(input->width, input->height);
        for (lsize_t i = 0; i < input->width; ++i) {
            for (lsize_t j = 0; j < input->height; ++j) {
                int8_t abs_v = (input->matrix[i][j] > 0) ? input->matrix[i][j] : 0;
                out->matrix[i][j] = abs_v;
            }
        }
        out->scale = input->scale;
        // Store activation and its exponent
        // free_m8(&layer->activations);
        // layer->activations = m_cpy(&result);
        m_cpy(&layer->activations, out);

        // print_layer(layer, "ReLU");
    } else {
        error("Error: Unsupported layer type in layer_forward.");
    }
}

__bank(2) Matrix8 network_forward_2(const Network* network, const Matrix8* X) {
    // // Matrix8 next_activations = {0};
    // Matrix8 next_activations = init_m8(0, 0);
    // layer_forward_1(network->layers[0], X, &next_activations);
    // Matrix8 result = next_activations;
    // for (lsize_t i = 1; i < network->num_layers; ++i) {
    //     layer_forward_1(network->layers[i], &next_activations, &result);
    //     next_activations = result;
    // }
    // return result;
    Matrix8 next_activations = {0};
    // printf("Layer: 0\n");
    layer_forward_1(network->layers[0], X, &next_activations);
    Matrix8 current_activations = next_activations; // Keep the latest (contains data and scale)
    print_matrix8_d(&current_activations, "First Inter activations");
    next_activations.width = 0;
    next_activations.matrix = NULL;
    for (lsize_t i = 1; i < network->num_layers; ++i) {
        // printf("Layer: %d\n", i);
        layer_forward_1(network->layers[i], &current_activations, &next_activations);
        free_m8(&current_activations); // Free intermediate activations
        current_activations = next_activations; // Keep the latest (contains data and scale)
        print_matrix8_d(&current_activations, "Inter activations");
        next_activations.width = 0;
        next_activations.matrix = NULL;
    }
    // The final current_activations is the output of the last layer
    return current_activations;
}

void assert_m8_equals(const Matrix8* a, const Matrix8* b) {
    if (a->width != b->width || a->height != b->height) {
        error("Error: Matrix dimensions do not match.\n");
        error("a shape: (%d, %d), b shape: (%d, %d)\n", a->width, a->height, b->width, b->height);
        exit(1);
    }

    for (lsize_t i = 0; i < a->width; ++i) {
        for (lsize_t j = 0; j < a->height; ++j) {
            if (a->matrix[i][j] != b->matrix[i][j]) {
                error("Error: Matrix values do not match at (%d, %d).\n", i, j);
                error("a value: %d, b value: %d\n", a->matrix[i][j], b->matrix[i][j]);
                exit(1);
            }
        }
    }
}

__bank(2) Matrix8 network_forward(const Network* network, const Matrix8* X) {
    return network_forward_2(network, X);
}

Matrix32 to32(const Matrix8* matrix) {
    Matrix32 result = init_m32(matrix->width, matrix->height);
    for (lsize_t i = 0; i < matrix->width; ++i) {
        for (lsize_t j = 0; j < matrix->height; ++j) {
            result.matrix[i][j] = (int32_t) matrix->matrix[i][j];
        }
    }
    result.scale = matrix->scale;
    return result;
}

// def StoShift(input, shift):
//     '''
//     Shift the input using
//     stochastic rounding
//     '''
//     tensor_type = input.dtype
//     round_temp = input//(2**shift)
//     prob = torch.abs(input - round_temp * (2**shift))
//     rand_num = torch.randint(low = 0, high=2**shift,size=prob.size(), dtype = tensor_type, device='cuda')
//     round_decision = torch.where(prob <= rand_num,
//                                  torch.tensor(0,dtype=tensor_type,device='cuda'),
//                                  torch.tensor(1,dtype=tensor_type,device='cuda'))
//     round_decision = round_decision * torch.sign(input)
//     return int8_clip(round_temp + round_decision)
Matrix8 sto_shift(const Matrix32* matrix, int8_t shift) {
    // Matrix32 temp = m64_to_m32(matrix);
    Matrix32 temp = init_m32(matrix->width, matrix->height);
    for (lsize_t i = 0; i < matrix->width; ++i) {
        for (lsize_t j = 0; j < matrix->height; ++j) {
            temp.matrix[i][j] = matrix->matrix[i][j];
        }
    }
    temp.scale = matrix->scale;
    Matrix8 result = init_m8(temp.width, temp.height);
    for (lsize_t i = 0; i < temp.width; ++i) {
        for (lsize_t j = 0; j < temp.height; ++j) {
            int32_t round_temp = temp.matrix[i][j] >> shift;
            int32_t prob = abs(temp.matrix[i][j] - (round_temp << shift));
            // int32_t rand_num = rand() % (1 << shift);
            // TODO Not Uniform Distribution
            int32_t rand_num = urand8() % (1 << shift);
            int32_t round_decision = (prob <= rand_num) ? 0 : 1;
            round_decision *= sign(temp.matrix[i][j]);
            if (round_temp + round_decision > 127) result.matrix[i][j] = 127;
            else if (round_temp + round_decision < -128) result.matrix[i][j] = -128;
            else result.matrix[i][j] = (int8_t) (round_temp + round_decision);
        }
    }
    #ifdef LIN_DEBUG
    println("STO Shift scales: %d %d", matrix->scale, shift);
    #endif
    result.scale = matrix->scale + shift;
    free_m32(&temp);
    return result;
}

/**
 * Compute integer "cross‐entropy" gradient as in your Python version.
 *
 * @param out_val        Flattened [batch × classes] logits (int64_t).
 * @param batch          Number of examples in the batch.
 * @param classes        Number of classes (size of logits vector per example).
 * @param out_exp        Integer exponent scalar.
 * @param target_indices Array of length batch: the true‐class index for each example.
 * @return               TiLossResult containing quantized out_grad and err_out_exp.
 */
 Matrix8 loss_gradient(const Matrix8* out_val, const Vector8* target_indices) {
    Matrix8 res;

    lsize_t classes = out_val->height;
    lsize_t batch = out_val->width;

    // 1) copy logits into a mutable Matrix64 `s`
    Matrix32 s = init_m32(batch, classes);
    s.scale = 0;
    for (lsize_t r = 0; r < s.width; ++r)
        for (lsize_t c = 0; c < s.height; ++c)
            s.matrix[r][c] = out_val->matrix[r][c];

    // 2) compute grad64 in a second Matrix64
    // Matrix64 grad64 = init_m64(classes, batch);

    if (out_val->scale > -7) {
        // change base from e to 2: multiply by 47274/(2^15)
        for (lsize_t r = 0; r < s.width; ++r) {
            for (lsize_t c = 0; c < s.height; ++c) {
                // s.matrix[r][c] = s.matrix[r][c] * 47274 / (1<<15);
                if (s.matrix[r][c] < 0) s.matrix[r][c] = -((-s.matrix[r][c] * 47274) >> 15);
                else s.matrix[r][c] = (s.matrix[r][c] * 47274) >> 15;
            }
        }

        if (out_val->scale >= 0) {
            int32_t m = 1 << out_val->scale;
            for (lsize_t r = 0; r < s.width; ++r) {
                for (lsize_t c = 0; c < s.height; ++c) {
                    s.matrix[r][c] *= m;
                }
        }
        } else {
            // int32_t d = 1 << -out_val->scale;
            for (lsize_t r = 0; r < s.width; ++r) {
                for (lsize_t c = 0; c < s.height; ++c) {
                    // s.matrix[r][c] /= d;
                    if (s.matrix[r][c] < 0) s.matrix[r][c] = -(-s.matrix[r][c] >> -out_val->scale);
                    else s.matrix[r][c] >>= -out_val->scale;
                }
            }
        }

        // subtract per‐row (max−10), clamp ≥0, then grad=2^s−1
        for (lsize_t r = 0; r < s.width; ++r) {
            int32_t mx = INT32_MIN;
            for (lsize_t c = 0; c < s.height; ++c)
                mx = max(mx, s.matrix[r][c]);
            int32_t off = mx - 10;
            for (lsize_t c = 0; c < s.height; ++c) {
                int32_t v = s.matrix[r][c] - off;
                s.matrix[r][c] = max(v, 0);
                // s.matrix[r][c] = (1 << s.matrix[r][c]) - 1;
                s.matrix[r][c] = (((int32_t) 1) << s.matrix[r][c]) - 1;
            }
        }

    } else {
        // small‐exp approx: 1+x+0.5x^2  ⇒ 2^(1−2e)+ s·2^(1−e)+s^2
        int32_t t1 = ((int32_t) 1) << (1 - 2*out_val->scale);
        int32_t t2 = ((int32_t) 1) << (1 - out_val->scale);
        for (lsize_t r = 0; r < s.width; ++r) {
            for (lsize_t c = 0; c < s.height; ++c) {
                s.matrix[r][c] = t1 + s.matrix[r][c] * t2 + s.matrix[r][c] * s.matrix[r][c];
            }
        }
    }

    // 3) row‐sums
    int32_t* row_sum = malloc(s.width * sizeof(int32_t));
    for (lsize_t r = 0; r < s.width; ++r) {
        int32_t sum = 0;
        for (lsize_t c = 0; c < s.height; ++c)
            sum += s.matrix[r][c];
        row_sum[r] = sum;
        // if (row_sum[r] > INT32_MAX ||  row_sum[r] < INT32_MIN) {
        //     log("%" PRId64, row_sum[r]);
        // }
    }

    // 4) normalize by 2^11/row_sum, then subtract row‐sum at target
    for (lsize_t r = 0; r < s.width; ++r) {
        // int32_t norm = 1 << 11;
        int32_t total = 0;
        for (lsize_t c = 0; c < s.height; ++c) {
            // s.matrix[r][c] = s.matrix[r][c] * norm / (row_sum[r] == 0 ? 1 : row_sum[r]);
            // s.matrix[r][c] = s.matrix[r][c] * norm / row_sum[r];
            s.matrix[r][c] = (s.matrix[r][c] << 11) / row_sum[r];
            total += s.matrix[r][c];
        }
        // if (total > INT32_MAX || total < INT32_MIN) log("%" PRId64, total);
        // subtract sum at the true‐class pos
        // if (target_indices->vector[r] >= s.height || target_indices->vector[r] < 0) exit(1);
        s.matrix[r][target_indices->vector[r]] -= total;
    }
    free(row_sum);
    // free_m64(&s);

    // for (lsize_t r = 0; r < s.width; ++r) {
    //     for (lsize_t c = 0; c < s.height; ++c) {
    //         if (s.matrix[r][c] > INT32_MAX || s.matrix[r][c] < INT32_MIN) {
    //             log("%" PRId64, s.matrix[r][c]);
    //         }
    //     }
    // }

    // 5) cast to int32 + StoShift
    // Matrix32 tmp = init_m32(classes, batch);
    // for (lsize_t r = 0; r < batch; ++r)
    //     for (lsize_t c = 0; c < classes; ++c)
    //         tmp.matrix[r][c] = (int32_t)grad64.matrix[r][c];
    
    print_matrix32_d(&s, "s");

    // res.out_grad = sto_shift(&s, 4);
    res = sto_shift(&s,4);
    // free_m32(&tmp);
    free_m32(&s);

    return res;
}

Matrix8 network_backward_1(const Network* network, const Vector8* Y) {
    const Layer* last_layer = network->layers[network->num_layers - 1];
    Matrix8 out_activations = last_layer->activations;

    Matrix8 loss = loss_gradient(&out_activations, Y);

    print_matrix8_d(&out_activations, "OutAct");
    print_vector8_d(Y, "Y");
    print_matrix8_d(&loss, "Loss");

    for (int i = network->num_layers - 1; i >= 0; --i) {
        layer_backward_1(network->layers[i], &loss, &loss);
    }
    free_m8(&loss);
    return loss;
}

Matrix8 network_backward(const Network* network, const Vector8* Y) {
    Matrix8 result = network_backward_1(network, Y);
    return result;
}

Matrix8 m_cpy_range(const Matrix8* orig, lsize_t start, lsize_t end) {
    Matrix8 result = init_m8(min(end, orig->width) - start + 1, orig->height);
    for (lsize_t i = 0; i < result.width; ++i) {
        for (lsize_t j = 0; j < result.height; ++j) {
            result.matrix[i][j] = orig->matrix[i + start][j];
        }
    }
    result.scale = orig->scale;
    return result;
}

Vector8 v_cpy_range(const Vector8* orig, lsize_t start, lsize_t end) {
    Vector8 result;
    result.length = end - start + 1;
    result.vector = malloc(result.length * sizeof(int8_t));
    result.scale = orig->scale;
    for (lsize_t i = 0; i < result.length; ++i) {
        result.vector[i] = orig->vector[i + start];
    }
    return result;
}

void m_cpy(Matrix8* to, const Matrix8 *m) {
    if (to->width != m->width || to->height != m->height) {
        free_m8(to);
        *to = init_m8(m->width, m->height);
        // for (lsize_t i = 0; i < to->width; ++i) {
        //     free(to->matrix[i]);
        // }
        // free(to->matrix);
        // to->width = m->width;
        // to->height = m->height;
        // to->matrix = malloc(m->width * sizeof(int8_t*));
        // for (lsize_t i = 0; i < m->width; ++i) {
        //     to->matrix[i] = malloc(m->height * sizeof(int8_t));
        // }
    }
    for (lsize_t i = 0; i < m->width; ++i) {
        for (lsize_t j = 0; j < m->height; ++j) {
            to->matrix[i][j] = m->matrix[i][j];
        }
    }
    to->scale = m->scale;
    // Matrix8 result = init_m8(m->width, m->height);
    // for (lsize_t i = 0; i < m->width; ++i) {
    //     for (lsize_t j = 0; j < m->height; ++j) {
    //         result.matrix[i][j] = m->matrix[i][j];
    //     }
    // }
    // result.scale = m->scale;
    // return result;
}

void free_network(Network* network) {
    for (uint8_t i = 0; i < network->num_layers; ++i) {
        free_layer(network->layers[i]);
    }
    free(network->layers);
    free(network);
}

void free_layer(Layer* layer) {
    if (!layer) return;
    if (layer->type == LINEAR) free_m8(&layer->weights);
    free_m8(&layer->activations);
    free_m8(&layer->input_copy);
    free(layer);
}

void print_layer(const Layer* layer, char* name) {
    println("[%s] Layer (%s) [%d -> %d]:", name, layer->type == LINEAR ? "Linear" : "ReLU", layer->activations.width, layer->activations.height);
    if (layer->type == LINEAR) print_matrix8(&layer->weights, "Weights");
    print_matrix8(&layer->activations, "Activations");
    // lsize_t zeros = 0;
    // for (lsize_t i = 0; i < layer->weights.width; ++i) {
    //     for (lsize_t j = 0; j < layer->weights.height; ++j) {
    //         if (layer->weights.matrix[i][j] == 0) zeros++;
    //     }
    // }
    // float ratio = (float) zeros / (float) (layer->weights.width * layer->weights.height);
    // log("Zeros: %.2f%% (%d/%d)", ratio * 100, zeros, layer->weights.width * layer->weights.height);
};

void print_network(const Network *network) {
    println("Network:");
    println("Batch Size: %d", network->layers[0]->activations.width);
    print("Input (%d) -> ", network->layers[0]->weights.height);
    for (lsize_t i = 0; i < network->num_layers; ++i) {
        print("%s", network->layers[i]->type == LINEAR ? "Linear" : "ReLU");
        if (network->layers[i]->type == LINEAR) {
            print(" (%d)", network->layers[i]->activations.height);
        }
        if (i < network->num_layers - 1) {
            print(" -> ");
        }
    }
    println();
}