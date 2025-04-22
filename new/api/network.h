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
    uint8_t num_layers;
} Network;

// --- Function Declarations ---

// Initialization
Network* init_network(uint8_t num_layers, uint8_t batch_size);
// Pass batch_size, input/output sizes, type
Layer* init_layer(uint8_t batch_size, uint8_t num_inputs, uint8_t num_neurons, LayerType type);
void free_network(Network* network);
void free_layer(Layer* layer);
void print_layer(Layer* layer, char* name);
void print_network(Network* network);

// Core Network Operations (Update signatures)
// Input matrix X should contain the *quantized* int8 data and its exponent
Matrix8 network_forward(Network* network, Matrix8* X); // Returns last layer's activations (Matrix8 includes scale)

// Takes target labels Y (int8_t, scale=0 assumed)
// Returns error propagated back to input (Matrix8 includes scale)
Matrix8 network_backward(Network* network, Vector8* Y);

// Layer Operations (Internal - Update signatures)
// Input matrix includes its exponent (scale)
// Returns output activation matrix (including its exponent)
Matrix8 layer_forward(Layer* layer, Matrix8* input);

// Input error matrix includes its exponent (scale)
// Returns error matrix for the previous layer (including its exponent)
Matrix8 layer_backward(Layer* layer, Matrix8* error_in);

// Weight Update (Internal - Update signature)
// Needs gradient (int8), gradient exponent, weight matrix (with its exponent)
void weight_update(Layer* layer, Matrix8* grad_int8, int8_t grad_exp);

// Activation/Error/Gradient Calculation (Internal - Update signatures)
// Returns Matrix8 containing value and exponent
Matrix8 act_calc(Matrix32* int32_acc, int8_t input_exp); // input_exp = combined exp of input and weight
// Returns Matrix8 containing value and exponent
Matrix8 err_calc(Matrix32* int32_acc, int8_t input_exp); // input_exp = combined exp of error and weight
// Returns Matrix8 for grad_int8 and the calculated shift amount
Matrix8 grad_calc(Matrix32* int32_acc, int8_t mu, int8_t* out_shift); // Pass mu, return shift via pointer

// Helper
void set_mu(int8_t value); // Function to set global mu if used

Matrix8 m_cpy_range(Matrix8* orig, uint8_t start, uint8_t end);
Vector8 v_cpy_range(Vector8* orig, uint8_t start, uint8_t end);
Matrix8 m_cpy(Matrix8* m);

#endif // NETWORK_H
