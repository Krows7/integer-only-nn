#ifndef NETWORK_H
#define NETWORK_H

#include "linear.h"

// --- Layer Types ---
typedef enum {
    LINEAR,
    RELU
} LayerType;

// --- Layer Structure ---
typedef struct Layer {
    LayerType type;
    Matrix8 weights;
    Matrix8 activations;
    Matrix8 input_copy;
} Layer;

// --- Network Structure ---
typedef struct {
    Layer** layers;
    lsize_t num_layers;
    lsize_t batch_size;
} Network;

// --- Function Declarations ---

void layer_backward_1(const Layer* layer, const Matrix8* input, const Matrix8* error_in, Matrix8* out);

__bank(1) Network* init_network(lsize_t num_layers, lsize_t batch_size);
__bank(1) Layer* init_layer(lsize_t batch_size, lsize_t num_inputs, lsize_t num_neurons, LayerType type);
__bank(1) void free_network(Network* network);
__bank(1) void free_layer(Layer* layer);
__bank(1) void print_layer(const Layer* layer, char* name);
__bank(1) void print_network(const Network* network);
__bank(1) Matrix8 loss_gradient(const Matrix8* out, const Vector8* Y);

// Core Network Operations (Update signatures)
// Input matrix X should contain the *quantized* int8 data and its exponent
__bank(2) Matrix8 network_forward(const Network* network, const Matrix8* X); // Returns last layer's activations (Matrix8 includes scale)

// Takes target labels Y (int8_t, scale=0 assumed)
// Returns error propagated back to input (Matrix8 includes scale)
__bank(2) Matrix8 network_backward(const Network* network, const Vector8* Y);

// Layer Operations (Internal - Update signatures)
// Input matrix includes its exponent (scale)
// Returns output activation matrix (including its exponent)
__bank(2) Matrix8 layer_forward(Layer* layer, const Matrix8* input);

// Input error matrix includes its exponent (scale)
// Returns error matrix for the previous layer (including its exponent)
__bank(2) Matrix8 layer_backward(const Layer* layer, const Matrix8* error_in);

// Weight Update (Internal - Update signature)
// Needs gradient (int8), gradient exponent, weight matrix (with its exponent)
__bank(2) void weight_update(const Layer* layer, const Matrix8* grad_int8);

// Activation/Error/Gradient Calculation (Internal - Update signatures)
// Returns Matrix8 containing value and exponent
__bank(2) Matrix8 act_calc(Matrix32* int32_acc, int8_t input_exp); // input_exp = combined exp of input and weight
// Returns Matrix8 containing value and exponent
__bank(2) Matrix8 err_calc(Matrix32* int32_acc); // input_exp = combined exp of error and weight
// Returns Matrix8 for grad_int8 and the calculated shift amount
__bank(2) Matrix8 grad_calc(Matrix32* int32_acc, int8_t mu, int8_t* out_shift); // Pass mu, return shift via pointer

// Helper
void set_mu(int8_t value); // Function to set global mu if used

__bank(2) Matrix8 m_cpy_range(const Matrix8* orig, lsize_t start, lsize_t end);
__bank(2) Vector8 v_cpy_range(const Vector8* orig, lsize_t start, lsize_t end);
// Matrix8 m_cpy(const Matrix8* m);
__bank(2) void m_cpy(Matrix8* to, const Matrix8* m);

#endif // NETWORK_H
