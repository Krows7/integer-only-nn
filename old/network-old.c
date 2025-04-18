#include <math.h>
#include "../src/api/linear-math.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

enum LayerKind {
    LINEAR,
    RELU
};

typedef struct Layer {
    enum LayerKind kind;
    Matrix8 weights;
    Matrix8 activations;
    // Matrix8 input; // Remove or rename this
    Matrix8 input_copy; // Add this field
    struct Layer* next_layer;
} Layer;
// typedef struct Layer {
//     enum LayerKind kind;
//     Matrix8 weights;
//     Matrix8 activations;
//     Matrix8 input;
//     struct Layer* next_layer;
// } Layer;

int8_t map(int x, int s) {
    return (x * 255) / (s - 1) - 128;
}

void fill_weight(Matrix8* weights) {
    int s = weights->width * weights->height;
    for (uint8_t i = 0; i < weights->width; ++i) {
        for (uint8_t j = 0; j < weights->height; ++j) {
            weights->matrix[i][j] = (int8_t) (rand() % 256 - 128);
            // weights->matrix[i][j] = map(i * weights->height + j, s - 1);
        }
    }
    weights->scale = -7;
}

Layer* init_layer(uint8_t batch_size, uint8_t num_inputs, uint8_t num_neurons, enum LayerKind kind, Layer* next_layer) {
    Layer* result = malloc(sizeof(Layer));
    result->kind = kind;
    result->weights = init_m8(num_neurons, num_inputs);
    result->activations = init_m8(batch_size, num_neurons);
    result->input_copy = init_m8(batch_size, num_inputs); // <<< ADD THIS LINE
    result->next_layer = next_layer;
    fill_weight(&result->weights);
    return result;
}
// Layer* init_layer(uint8_t num_inputs, uint8_t num_neurons, enum LayerKind kind, Layer* next_layer) {
//     Layer* result = malloc(sizeof(Layer));
//     result->kind = kind;
//     result->weights = init_m8(num_neurons, num_inputs);
//     result->activations = init_m8(1, num_neurons);
//     result->next_layer = next_layer;
//     fill_weight(&result->weights);
//     return result;
// }

void print_layer(Layer* layer, char* name) {
    printf("[%s] Layer:\n", name);
    print_matrix8(&layer->weights, "Weights");
    print_matrix8(&layer->activations, "Activations");
};

void free_layer(Layer* layer) {
    free_m8(&layer->weights);
    free_m8(&layer->activations);
    free(layer);
}

// ------------ Rounding ------------

int8_t psto_round(int32_t num, int8_t bp) {
    int32_t round_temp = num >> bp;
    int32_t prob = abs(num - (round_temp << bp));
    int32_t quantized_prob = prob >> (bp >> 1);
    int32_t pseudo_rand_num = prob - (quantized_prob << (bp >> 1));

    if ((bp & 1) != 0) pseudo_rand_num <<= 1;

    // if (quantized_prob <= pseudo_rand_num) return round_temp * sign(num);
    // return (round_temp + 1) * sign(num);
    if (quantized_prob <= pseudo_rand_num) return round_temp;
    return round_temp + sign(num);
}

void shift_and_round(Matrix32* matrix, Matrix8* out, uint8_t bp) {
    if (matrix->width != out->width || matrix->height != out->height) {
        fprintf(stderr, "Error in shift_and_round: Matrix dimensions do not match.\n");
        fprintf(stderr, "Matrix: (%d, %d), out: (%d, %d)\n",
                matrix->width, matrix->height, out->width, out->height);
        exit(1);
    }
    if (bp > 0) {
        for (uint8_t i = 0; i < matrix->width; ++i) {
            for (uint8_t j = 0; j < matrix->height; ++j) {
                // Optionally, add prints inside the shift_and_round loop:
                // Inside shift_and_round, inside the loops:
                int8_t rounded_value = psto_round(matrix->matrix[i][j], bp);
                // printf("  psto_round(%d, %d) -> %d\n", matrix->matrix[i][j], bp, rounded_value); // Add this
                out->matrix[i][j] = rounded_value;
                // out->matrix[i][j] = psto_round(matrix->matrix[i][j], bp);
            }
        }
    } else {
        for (uint8_t i = 0; i < matrix->width; ++i) {
            for (uint8_t j = 0; j < matrix->height; ++j) {
                out->matrix[i][j] = (int8_t) matrix->matrix[i][j];
            }
        }
    }
    out->scale = matrix->scale + bp;
}

Matrix8 get_shift_and_round_bp(Matrix32* matrix, uint8_t bp) {
    Matrix8 result = init_m8(matrix->width, matrix->height);
    shift_and_round(matrix, &result, bp);
    return result;
}

Matrix8 get_shift_and_round(Matrix32* matrix) {
    uint8_t b = effective_bitwidth(matrix);
    return get_shift_and_round_bp(matrix, max(0, b - 7));
}

Matrix8 to8(Matrix32* matrix) {
    Matrix8 result = init_m8(matrix->width, matrix->height);
    for (uint8_t i = 0; i < matrix->width; ++i) {
        for (uint8_t j = 0; j < matrix->height; ++j) {
            result.matrix[i][j] = (int8_t) matrix->matrix[i][j];
        }
    }
    result.scale = matrix->scale;
    return result;
}

int32_t abs_max32(Matrix32* matrix) {
    int32_t result = 0;
    for (uint8_t i = 0; i < matrix->width; ++i) {
        for (uint8_t j = 0; j < matrix->height; ++j) {
            if (abs(matrix->matrix[i][j]) > result) {
                result = abs(matrix->matrix[i][j]);
            }
        }
    }
    return result;
}

int8_t tensor_bitwidth(int32_t value) {
    return (int) ceil(log2((double) value));
}

int8_t range_estimate(Matrix32* matrix) {
    int32_t max_abs = abs_max32(matrix);
    if (max_abs == 0) return 0;
    return tensor_bitwidth(max_abs);
}

Matrix8 act_calc(Matrix32* temp) {
    // int32_bitwidth = RangeEstimate(int32_acc)
    // shift = int32_bitwidth-BITWIDTH
    // if shift > 0:
    //     exp_out = exp_in+shift
    //     temp = ACT_ROUND_METHOD(int32_acc, shift)
    // else:
    //     exp_out=exp_in
    //     temp = int32_acc.type(torch.int8)

    // return temp, exp_out
    int8_t b = range_estimate(temp);
    int8_t shift = b - 7;
    if (shift > 0) {
        return get_shift_and_round_bp(temp, shift);
    }
    return to8(temp);
}

Matrix8 layer_forward(Layer* layer, Matrix8* input) {
    // Check against input_copy dimensions
    if (layer->input_copy.width != input->width || layer->input_copy.height != input->height) {
         fprintf(stderr, "FATAL Error: Input dimension mismatch for layer input_copy buffer.\n");
         fprintf(stderr, "Expected (%d, %d), Got (%d, %d)\n",
                 layer->input_copy.width, layer->input_copy.height,
                 input->width, input->height); // Add more detail
         exit(1);
    }

    // Copy into input_copy
    for(uint8_t i = 0; i < input->width; ++i) {
        memcpy(layer->input_copy.matrix[i], input->matrix[i], input->height * sizeof(int8_t));
    }
    layer->input_copy.scale = input->scale;

    if (layer->kind == LINEAR) {
        // Use input_copy for multiplication
        Matrix32 temp = get_mul8_2t(&layer->input_copy, &layer->weights);
        Matrix8 result = act_calc(&temp);
        free_m32(&temp); // Keep the memory leak fix
        return result;
    } else if (layer->kind == RELU) {
        Matrix8 result = init_m8(input->width, input->height);
        for (uint8_t i = 0; i < input->width; ++i) {
            for (uint8_t j = 0; j < input->height; ++j) {
                result.matrix[i][j] = max(0, input->matrix[i][j]);
            }
        }
        result.scale = input->scale;
        return result;
    } else {
        fprintf(stderr, "Unknown layer kind: %d\n", layer->kind);
        exit(1);
    }
}
// Matrix8 layer_forward(Layer* layer, Matrix8* input) {
//     layer->input = input;
//     Matrix32 temp = get_mul8_2t(input, &layer->weights);
//     return act_calc(&temp);
// }

Matrix8 err_calc(Matrix32* temp) {
    // int32_bitwidth = RangeEstimate(int32_acc)
    // shift = int32_bitwidth-BITWIDTH
    // if shift > 0:
    //     temp =ERROR_ROUND_METHOD(int32_acc, shift)
    //     exp_out = shift
    // else:
    //     temp = int32_acc.type(torch.int8)
    //     exp_out= 0

    // return temp, exp_out
    int8_t b = range_estimate(temp);
    int8_t shift = b - 7;
    Matrix8 result;
    if (shift > 0) {
        Matrix8 result = get_shift_and_round_bp(temp, shift);
    } else {
        Matrix8 result = to8(temp);
    }
    result.scale -= temp->scale;
    return result;
}

int8_t mu = 5;

void set_mu(int8_t new_mu) {
    mu = new_mu;
}

Matrix8 zeros(uint8_t width, uint8_t height) {
    Matrix8 result = init_m8(width, height);
    for (uint8_t i = 0; i < width; ++i) {
        for (uint8_t j = 0; j < height; ++j) {
            result.matrix[i][j] = 0;
        }
    }
    return result;
}

Matrix8 grad_calc(Matrix32* temp) {
    // int32_bitwidth = RangeEstimate(int32_acc)
    // shift = int32_bitwidth-mu
    // if int32_bitwidth == 0:
    //     return Int8zeros(int32_acc.size()), 0
    // elif shift < 1:
    //     return int32_acc.type(torch.int8), 0
    // else:
    //     return GRAD_ROUND_METHOD(int32_acc,int32_bitwidth-mu), shift
    int8_t b = range_estimate(temp);
    int8_t shift = b - mu;
    if (b == 0) return zeros(temp->width, temp->height);
    else if (shift < 1) {
        Matrix8 result = to8(temp);
        result.scale = 0;
        return result;
    }
    Matrix8 result = get_shift_and_round_bp(temp, shift);
    result.scale -= temp->scale;
    return result;
}

// def weight_update(self):
//     p = self.weight
//     """ vanilla SGD """
//     self.grad, grad_shift = grad_calc(self.grad_int32acc, GRAD_BITWIDTH)
//     self.grad_exp = self.err_exp + grad_shift + self.act_in_exp
//     p.data = int8_clip(p.type(torch.int16)-self.grad.type(torch.int16))
void weight_update(Layer* layer, Matrix32* grad32, Matrix8* err) {
    Matrix8 grad = grad_calc(grad32);
    for (uint8_t i = 0; i < layer->weights.width; ++i) {
        for (uint8_t j = 0; j < layer->weights.height; ++j) {
            int16_t temp = (int16_t) layer->weights.matrix[i][j] - (int16_t) grad.matrix[i][j];
            if (temp > 127) temp = 127;
            else if (temp < -128) temp = -128;
            layer->weights.matrix[i][j] = (int8_t) temp;
        }
    }
    free_m8(&grad); // <<< FIX MEMORY LEAK 4
}

Matrix8 layer_backward(Layer* layer, Matrix8* e_in) {
    // err_in, self.err_exp = input
    // act, self.act_in_exp = self.act_in

    // err_out_int32 = int8mm(err_in, self.weight)
    // err_out, shift_bits = err_calc(err_out_int32)
    // self.err_exp += (shift_bits + self.weight_exp)

    // self.grad_int32acc = int8mm(err_in.transpose(0,1).contiguous(), act)
    // self.weight_update()

    // return err_out, self.err_exp

    if (layer->kind == LINEAR) {
        Matrix32 err_out_int32 = get_mul8(e_in, &layer->weights);
        Matrix8 err_out = err_calc(&err_out_int32);
        err_out.scale += layer->weights.scale + e_in->scale;
        Matrix32 grad_int32acc = get_mul8_1t(e_in, &layer->input_copy);
        weight_update(layer, &grad_int32acc, &err_out);

        free_m32(&err_out_int32); // <<< FIX MEMORY LEAK 2
        free_m32(&grad_int32acc); // <<< FIX MEMORY LEAK 3

        return err_out;
    } else if (layer->kind == RELU) {
        Matrix8 err_out = init_m8(e_in->width, e_in->height);
        for (uint8_t i = 0; i < e_in->width; ++i) {
            for (uint8_t j = 0; j < e_in->height; ++j) {
                err_out.matrix[i][j] = e_in->matrix[i][j] * (layer->input_copy.matrix[i][j] > 0);
            }
        }
        err_out.scale = e_in->scale;
        return err_out;
    } else {
        fprintf(stderr, "Unknown layer kind: %d\n", layer->kind);
        exit(1);
    }
}

// ------------ Loss Function ------------

// Cross-Entropy Loss
// t_i = e^(a_i * 2^S_a)
// e_i = 1/C(e^(a_i * 2^S_a) - y_i * C) = 1/C(t_i - y_i * C)
// C = \sum t_i
// Matrix8 loss_gradient(Layer* layer, Vector8* y) {
//     Matrix32 t = init_m32(1, layer->activations.height);
//     if (layer->activations.scale <= -7) {
//         // e^x = 1 + x + 1/2*x^2
//         // t_i = 1 + a_i * 2^S_a + 1/2(a_i)^2 * (2^S_a)^2 = 2^(2*S_a - 1) * (2 ^ (1 - 2*S_a) + a_i * 2 ^ (1 - S_a) + (a_i)^2)
//         int32_t t_const_1 = 2 << (1 - 2 * layer->activations.scale);
//         int32_t t_const_2 = 2 << (1 - layer->activations.scale);
//         for (uint8_t i = 0; i < layer->activations.height; ++i) {
//             int8_t a = layer->activations.matrix[0][i];
//             t.matrix[0][i] = t_const_1 * (t_const_2 + a * t_const_2 + a * a);
//         }
//     } else {
//         int32_t t_max = INT32_MIN;
//         for (uint8_t i = 0; i < layer->activations.height; ++i) {
//             t.matrix[0][i] = (layer->activations.matrix[0][i] * 47274) >> 15;
//             if (layer->activations.scale >= 0) t.matrix[0][i] <<= layer->activations.scale;
//             else t.matrix[0][i] >>= -layer->activations.scale;
//             if (t.matrix[0][i] > t_max) t_max = t.matrix[0][i];
//         }
//         int32_t offset = t_max - 10;
//         for (uint8_t i = 0; i < layer->activations.height; ++i) {
//             t.matrix[0][i] -= offset;
//             t.matrix[0][i] = max(0, t.matrix[0][i]);
//             t.matrix[0][i] = (2 << t.matrix[0][i]) - 1;
//         }
//     }
//     int32_t C = 0;
//     for (uint8_t i = 0; i < layer->activations.height; ++i) {
//         C += t.matrix[0][i];
//     }

//     const int N_SCALE_BITS = 15;
//     const int32_t scale_factor = 1 << N_SCALE_BITS;

//     if (C == 0) { // Avoid division by zero
//         // Handle this case, maybe set gradient to zero or some error state
//         // For now, just zero out t
//         for (uint8_t i = 0; i < layer->activations.height; ++i) {
//             t.matrix[0][i] = 0;
//         }
//     } else {
//         for (uint8_t i = 0; i < layer->activations.height; ++i) {
//             // Calculate p_i scaled by scale_factor: (t_i * scale_factor) / C
//             int64_t scaled_p_i_num = (int64_t)t.matrix[0][i] * scale_factor; // Use 64-bit intermediate
//             int32_t scaled_p_i = (int32_t)(scaled_p_i_num / C);

//             // Calculate y_i scaled by scale_factor
//             int32_t scaled_y_i = (i == y->vector[0]) ? scale_factor : 0;

//             // Gradient g_i = p_i - y_i (scaled)
//             t.matrix[0][i] = scaled_p_i - scaled_y_i;
//         }
//     }
//     // The resulting gradient 't' is now scaled by N_SCALE_BITS (e.g., 15)
//     // Update the scale field if necessary, though shift_round_bp uses its own logic
//     // t.scale = N_SCALE_BITS; // Or adjust based on original t.scale if it existed

//     // --- End of replacement ---

//     // Final quantization (keep this, but potentially adjust mu as per Solution 1)
//     const uint8_t loss_grad_shift = 1; // Example: Try no shift
//     // const uint8_t loss_grad_shift = max(0, N_SCALE_BITS - 7); // e.g., 15 - 7 = 8

//     Matrix8 e = get_shift_and_round_bp(&t, loss_grad_shift); // Apply the calculated shift
//     // The scale of 'e' will be set to loss_grad_shift by shift_and_round

//     // Inside loss_gradient, before the return statement:
//     // printf("Intermediate t: [%d, %d, %d]\n", t.matrix[0][0], t.matrix[0][1], t.matrix[0][2]);
//     // printf("Calculated e before return: [%d, %d, %d], scale=%d\n", e.matrix[0][0], e.matrix[0][1], e.matrix[0][2], e.scale);
//     free_m32(&t); // Free the intermediate t matrix
//     return e;

//     // Matrix8 e = init_m8(1, layer->activations.height);
//     // for (uint8_t i = 0; i < layer->activations.height; ++i) {
//     //     // t.matrix[0][i] = t.matrix[0][i] * (2 << 11) / C;
//     //     t.matrix[0][i] = t.matrix[0][i] / C;
//     // }
//     // // C = 0;
//     // // for (uint8_t i = 0; i < layer->activations.height; ++i) {
//     // //     C += t.matrix[0][i];
//     // // }
//     // t.matrix[0][y->vector[0]] -= C;
//     // // for (uint8_t i = 0; i < layer->activations.height; ++i) {
//     // //     e.matrix[0][i] = (int8_t) ((t.matrix[0][i] - y->vector[i] * C) / C);
//     // // }
//     // e = shift_round_bp(&t, mu);
//     // return e;
// }

Matrix32 to32(Matrix8* matrix) {
    Matrix32 result = init_m32(matrix->width, matrix->height);
    for (uint8_t i = 0; i < matrix->width; ++i) {
        for (uint8_t j = 0; j < matrix->height; ++j) {
            result.matrix[i][j] = (int32_t) matrix->matrix[i][j];
        }
    }
    result.scale = matrix->scale;
    return result;
}

Matrix64 to64(Matrix8* matrix) {
    Matrix64 result = init_m64(matrix->width, matrix->height);
    for (uint8_t i = 0; i < matrix->width; ++i) {
        for (uint8_t j = 0; j < matrix->height; ++j) {
            result.matrix[i][j] = (int64_t) matrix->matrix[i][j];
        }
    }
    result.scale = matrix->scale;
    return result;
}

Matrix64 get_sum(Matrix64* matrix) {
    Matrix64 result = init_m64(matrix->width, 1);
    for (uint8_t i = 0; i < matrix->width; ++i) {
        for (uint8_t j = 0; j < matrix->height; ++j) {
            result.matrix[i][0] += matrix->matrix[i][j];
        }
    }
    return result;
}

Matrix32 m64_to_m32(Matrix64* matrix) {
    Matrix32 result = init_m32(matrix->width, matrix->height);
    for (uint8_t i = 0; i < matrix->width; ++i) {
        for (uint8_t j = 0; j < matrix->height; ++j) {
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
Matrix8 sto_shift(Matrix64* matrix, int8_t shift) {
    Matrix32 temp = m64_to_m32(matrix);
    Matrix8 result = init_m8(temp.width, temp.height);
    for (uint8_t i = 0; i < temp.width; ++i) {
        for (uint8_t j = 0; j < temp.height; ++j) {
            int32_t round_temp = temp.matrix[i][j] >> shift;
            int32_t prob = abs(temp.matrix[i][j] - (round_temp << shift));
            int32_t rand_num = rand() % (1 << shift);
            int32_t round_decision = (prob <= rand_num) ? 0 : 1;
            round_decision *= sign(temp.matrix[i][j]);
            if (round_temp + round_decision > 127) result.matrix[i][j] = 127;
            else if (round_temp + round_decision < -128) result.matrix[i][j] = -128;
            else result.matrix[i][j] = round_temp + round_decision;
        }
    }
    result.scale = matrix->scale + shift;
    return result;
}

// class TiLoss(Module):
//     def forward(self, out_val, out_exp, target):
//         err_out_exp=0
//         # integer cross entropy loss
//         s=out_val.type(torch.int64)
//         if out_exp >-7:
//             # if out_exp is big enough
//             # change the base in log softmax from e to 2
//             # to approx integer loss
//             s=s*47274/(2**15)
//             if out_exp>=0:
//                 s=s*2**out_exp
//             else:
//                 s=s/(2**-out_exp)

//             out_max, _ = torch.max(s,dim=1)
//             offset = out_max-10
//             s=s-offset.view(-1,1)
//             s=torch.max(s,Int8Tensor(0).type(torch.int64))
//             out_grad = 2**s-1
//         else:
//             # if out_exp is too small s will be all 0
//             # use another apporximation 1+e^x = 1 + x + 0.5 x^2 + o(x^2)
//             out_grad = 2**(1-2*out_exp.type(torch.int64)) + \
//                 s*2**(1-out_exp.type(torch.int64)) + s*s

//         out_sum = out_grad.sum(1,dtype=torch.int64)

//         out_grad = out_grad*(2**11)/out_sum.view(-1,1)
//         out_grad[torch.arange(out_val.size(0)), target] -= out_grad.sum(1,dtype=torch.int64)
//         self.out_grad = StoShift(out_grad.type(torch.int32),4)

//         return self.out_grad, err_out_exp
Matrix8 loss_gradient(Matrix8* out, Vector8 *y) {
    Matrix64 grad;
    Matrix64 t, mx, sum; // Declare intermediates needed for freeing
    int used_t = 0, used_mx = 0, used_sum = 0; // Track which were allocated
    if (out->scale <= -7) {
        t = to64(out); used_t = 1;
        for (uint8_t i = 0; i < out->width; ++i) {
            for (uint8_t j = 0; j < out->height; ++j) {
                t.matrix[i][j] = t.matrix[i][j] * 47274 / (1 << 15);
                if (out->scale >= 0) t.matrix[i][j] <<= out->scale;
                else t.matrix[i][j] >>= -out->scale;
            }
        }
        mx = init_m64(out->width, 1); used_mx = 1;
        for (uint8_t i = 0; i < out->width; ++i) {
            mx.matrix[i][0] = t.matrix[i][0];
            for (uint8_t j = 1; j < out->height; ++j) {
                if (t.matrix[i][j] > mx.matrix[i][0]) mx.matrix[i][0] = t.matrix[i][j];
            }
        }
        grad = init_m64(out->width, out->height);
        for (uint8_t i = 0; i < out->width; ++i) {
            for (uint8_t j = 0; j < out->height; ++j) {
                t.matrix[i][j] -= mx.matrix[i][0] - 10;
                t.matrix[i][j] = max(0, t.matrix[i][j]);
                grad.matrix[i][j] = (1 << t.matrix[i][j]) - 1;
            }
        }
    } else {
        grad = init_m64(out->width, out->height);
        for (uint8_t i = 0; i < out->width; ++i) {
            for (uint8_t j = 0; j < out->height; ++j) {
                grad.matrix[i][j] = (1 << (1 - 2 * out->scale)) + out->matrix[i][j] * (1 << (1 - out->scale)) + out->matrix[i][j] * out->matrix[i][j];
            }
        }
    }
    sum = get_sum(&grad); used_sum = 1;
    for (uint8_t i = 0; i < out->width; ++i) {
        for (uint8_t j = 0; j < out->height; ++j) {
            grad.matrix[i][j] = grad.matrix[i][j] * (1 << 11) / sum.matrix[i][0];
        }
    }
    sum = get_sum(&grad);
    for (uint8_t i = 0; i < out->width; ++i) {
        grad.matrix[i][y->vector[i]] -= sum.matrix[i][0];
    }
    // print_matrix32(&grad, "grad");
    // Matrix8 result = get_shift_and_round_bp(&grad, 4);
    Matrix8 result = sto_shift(&grad, 4);
    result.scale = 0;

    // Free intermediates
    if(used_t) free_m64(&t);
    if(used_mx) free_m64(&mx);
    if(used_sum) free_m64(&sum); // Free the last 'sum' allocated
    // Need to free the other 'sum' if recalculated
    free_m64(&grad); // Free the main grad Matrix32

    return result;
}

typedef struct Network {
    Layer** layers;
    uint8_t num_layers;
    uint8_t batch_size;
} Network;

Network* init_network(uint8_t num_layers, uint8_t batch_size) {
    Network* result = malloc(sizeof(Network));
    result->layers = malloc(num_layers * sizeof(Layer));
    result->num_layers = num_layers;
    result->batch_size = batch_size;
    return result;
}

void free_network(Network* network) {
    for (uint8_t i = 0; i < network->num_layers; ++i) {
        free_layer(network->layers[i]);
    }
    free(network->layers);
    free(network);
}

Matrix8 network_forward(Network* network, Matrix8* input) {
    Matrix8 current_input = *input; // Start with original input
    Matrix8 next_input;             // To hold result from layer_forward
    int is_first_layer = 1;

    for (uint8_t i = 0; i < network->num_layers; ++i) {
        next_input = layer_forward(network->layers[i], &current_input); // Pass current, get next

        // Free the previous layer's output (which was current_input)
        // UNLESS it was the original input passed to the function
        if (!is_first_layer) {
            free_m8(&current_input);
        }
        is_first_layer = 0;
        current_input = next_input; // The new output becomes the input for the next iteration
    }
    // The final 'current_input' is the network's output
    return current_input;
}
// Matrix8 network_forward(Network* network, Matrix8* input) {
//     Matrix8 result = layer_forward(network->layers[0], input);
//     for (uint8_t i = 1; i < network->num_layers; ++i) {
//         result = layer_forward(network->layers[i], &result);
//     }
//     return result;
// }

Vector8 toVector(Matrix8 matrix) {
    Vector8 result;
    result.vector = malloc(matrix.height * sizeof(int8_t));
    for (uint8_t i = 0; i < matrix.height; ++i) {
        result.vector[i] = matrix.matrix[0][i];
    }
    return result;
}

Matrix8 network_backward(Network* network, Vector8* y) {
    // Vector8 y = toVector(e_in);
    Matrix8 x = loss_gradient(&network->layers[network->num_layers - 1]->activations, y);
    for (uint8_t i = network->num_layers - 1; i > 0; --i) {
        x = layer_backward(network->layers[i], &x);
    }
    return x;
}

#define min(a, b) ((a) < (b) ? (a) : (b))

Matrix8 m_cpy_range(Matrix8* orig, uint8_t start, uint8_t end) {
    Matrix8 result = init_m8(min(end, orig->width) - start + 1, orig->height);
    for (uint8_t i = 0; i < result.width; ++i) {
        for (uint8_t j = 0; j < result.height; ++j) {
            result.matrix[i][j] = orig->matrix[i + start][j];
        }
    }
    result.scale = orig->scale;
    return result;
}

Vector8 v_cpy_range(Vector8* orig, uint8_t start, uint8_t end) {
    Vector8 result;
    result.vector = malloc((end - start + 1) * sizeof(int8_t));
    for (uint8_t i = 0; i < end - start + 1; ++i) {
        result.vector[i] = orig->vector[i + start];
    }
    return result;
}