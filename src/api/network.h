#ifndef NETWORK_H
#define NETWORK_H

#include "linear-math.h"
#include "quantization.h" // Or quantization.h

// Add near the top of network.c or in network.h
#define DEBUG_LOG_LEVEL 0 // 0 = Off, 1 = Basic Logs, 2 = Verbose Logs

// --- Layer Types ---
typedef enum {
    LINEAR,
    RELU // Keep ReLU definition even if unused in the best run
    // Add other types like FLATTEN if needed
} LayerType;

// --- Layer Structure ---
typedef struct Layer {
    LayerType type;
    Matrix8 weights; // Includes weight exponent (scale)
    Matrix8 activations; // Includes activation exponent (scale)
    Matrix8 input_copy; // Includes input exponent (scale) - Store input *before* quantization? No, store quantized input.
    // Add fields to store exponents explicitly if needed, though Matrix8.scale should suffice
    int8_t act_exp; // Store activation exponent explicitly
    int8_t input_exp; // Store input exponent explicitly

    // Pointers to next/prev layers might be useful but seem unused in original code
    struct Layer* next_layer; // Example

    // Layer-specific parameters (e.g., stride, padding for Conv/Pool)
    // ...

} Layer;

// --- Network Structure ---
typedef struct {
    Layer** layers;
    uint8_t num_layers;
    uint8_t batch_size;
    // Add fields to store intermediate exponents if needed
} Network;

// --- Function Declarations ---

// Initialization
Network* init_network(uint8_t num_layers, uint8_t batch_size);
// Pass batch_size, input/output sizes, type, and potentially next layer pointer
Layer* init_layer(uint8_t batch_size, uint16_t num_inputs, uint16_t num_neurons, LayerType type, Layer* next_layer);
void free_network(Network* network);
void free_layer(Layer* layer);

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

void print_layer(Layer* layer, char* name);

#endif // NETWORK_H
