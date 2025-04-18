#include "../api/network.h" // Includes network logic and implicitly linear-math
#include <stdint.h>
#include <stdio.h>          // For file I/O, printf
#include <stdlib.h>         // For atoi, malloc, free, exit, rand, srand
#include <string.h>         // For strtok, memcpy, memset
#include <time.h>           // For seeding random weights
#include <limits.h>         // For INT_MAX, INT_MIN (used in prediction)
// #include <linux/limits.h> // Original include, likely not needed now
#include "../api/weights.h"

// --- Dataset Constants ---
#define INPUT_SIZE 64       // 8x8 grid -> 64 features
#define NUM_CLASSES 10      // Digits 0-9
#define MAX_LINE_LENGTH 512 // Max characters per line in data files

// --- Data File Paths (Adjust if necessary) ---
#define TRAIN_DATA_FILE "/home/huy/digits/optdigits.tra"
#define TEST_DATA_FILE "/home/huy/digits/optdigits.tes"
// Note: The dataset description says 3823 training, 1797 testing.
// We'll read until EOF or max samples reached.
#define MAX_TRAIN_SAMPLES 3823
#define MAX_TEST_SAMPLES 1797
#define BATCH_SIZE 32

// --- Network Hyperparameters ---
#define HIDDEN_NEURONS 64   // Example size - adjust as needed
#define NUM_EPOCHS 5       // Increase epochs for better learning
#define LEARNING_RATE_MU 4  // Corresponds to 'mu' in network.c grad_calc (default was 4)

// --- Helper Function: Scale pixel value (0-16) to int8_t range ---
int8_t scale_pixel(int pixel_value) {
    // Linear scaling: map 0..16 to approx -127..+127
    // Formula: (value / max_value) * range - offset
    // (pixel_value / 16.0) * 254.0 - 127.0
    // Use integer arithmetic carefully to avoid floating point
    // Scaled = (pixel * 254) / 16 - 127
    // Scaled = (pixel * 127) / 8 - 127
    int32_t scaled = ((int32_t)pixel_value * 127) / 8 - 127;

    // Clamp to int8_t range
    if (scaled > 127) return 127;
    if (scaled < -128) return -128;
    return (int8_t)scaled;
}

// --- Helper Function: Load Optdigits Data ---
int load_optdigits_data(const char* filename, Matrix8* X, Vector8* Y, int max_samples) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening data file");
        fprintf(stderr, "Failed to open: %s\n", filename);
        exit(1);
    }

    char line[MAX_LINE_LENGTH];
    int sample_count = 0;
    const char* delim = ",";

    while (fgets(line, sizeof(line), file) && sample_count < max_samples) {
        if (strlen(line) < INPUT_SIZE) continue; // Skip potentially empty/short lines

        char* token;
        int feature_count = 0;

        // Read features (first 64 values)
        token = strtok(line, delim);
        while (token != NULL && feature_count < INPUT_SIZE) {
            int pixel_value = atoi(token);
            X->matrix[sample_count][feature_count] = scale_pixel(pixel_value);
            feature_count++;
            token = strtok(NULL, delim);
        }

        // Read the label (the last value)
        if (token != NULL && feature_count == INPUT_SIZE) {
            Y->vector[sample_count] = (int8_t)atoi(token); // Label is 0-9
        } else {
            fprintf(stderr, "Warning: Malformed line or incorrect feature count at sample %d in %s\n", sample_count, filename);
            // Optionally skip this sample or handle error differently
            continue; // Skip this sample
        }

        sample_count++;
    }

    fclose(file);

    // Update matrix/vector dimensions to actual loaded count
    X->width = sample_count;
    Y->length = sample_count;

    // Set scale factor for input data (after scaling, values are direct int8_t)
    X->scale = 0;
    Y->scale = 0; // Labels don't typically have a scale in this context

    printf("Loaded %d samples from %s.\n", sample_count, filename);
    return sample_count;
}

// --- Helper Function: Predict Class ---
Vector8* predict(Network* network, Matrix8* x_sample) {
    Matrix8 output_activations = network_forward(network, x_sample);

    Vector8* result = malloc(sizeof(Vector8));
    result->length = x_sample->width;
    result->vector = malloc(result->length * sizeof(int8_t));
    
    // Find the index of the maximum activation in the output layer
    // int predicted_class = -1;
    
    // output_activations has shape (1, NUM_CLASSES)
    // if (output_activations.width != 1 || output_activations.height != NUM_CLASSES) {
    //     fprintf(stderr, "Error: Unexpected output activation dimensions (%d, %d)\n",
    //         output_activations.width, output_activations.height);
    //         // Don't free output_activations here if it points to internal network state
    //         return NULL; // Indicate error
    //     }
        
    for (uint8_t k = 0; k < result->length; ++k) {
        int8_t max_activation = -128; // Smallest int8_t value
        for (uint8_t i = 0; i < output_activations.height; ++i) {
            if (output_activations.matrix[k][i] > max_activation) {
                max_activation = output_activations.matrix[k][i];
                result->vector[k] = i;
            }
        }
    }

    // IMPORTANT: network_forward in the provided network.c seems to return
    // a pointer to the *last layer's activation matrix*.
    // DO NOT free output_activations here, as it's part of the network state.
    // free_m8(&output_activations); // <-- DO NOT DO THIS

    return result;
}

// --- Helper Function: Evaluate Network ---
void evaluate(Network* network, Matrix8* X_test, Vector8* Y_test, int num_test_samples) {
    printf("\n--- Evaluating Network ---\n");
    if (num_test_samples == 0) {
        printf("No test samples loaded for evaluation.\n");
        return;
    }

    int correct_predictions = 0;
    Matrix8 x_sample = init_m8(1, INPUT_SIZE); // Holder for one test sample

    for (int sample_idx = 0; sample_idx < num_test_samples && sample_idx + BATCH_SIZE <= num_test_samples; sample_idx += BATCH_SIZE) {
        // Copy the current sample data into the single-sample holders

        Matrix8 x_sample = m_cpy_range(X_test, sample_idx, sample_idx + BATCH_SIZE - 1);
        Vector8 y_sample = v_cpy_range(Y_test, sample_idx, sample_idx + BATCH_SIZE - 1);
    // for (int i = 0; i < num_test_samples; ++i) {
    //     // Copy test sample data
    //     memcpy(x_sample.matrix[0], X_test->matrix[i], INPUT_SIZE * sizeof(int8_t));
    //     x_sample.scale = X_test->scale; // Copy scale

        Vector8* predicted = predict(network, &x_sample);
        

        for (uint8_t k = 0; k < x_sample.width; ++k) {
            if (predicted->vector[k] == y_sample.vector[k]) {
                correct_predictions++;
            }
        }

        free_v8(predicted);
        
        // int predicted = predict(network, &x_sample);
        // int actual = Y_test->vector[i];

        // if (predicted == actual) {
        //     correct_predictions++;
        // }
        // Optional: Print prediction vs actual for debugging
        // if (i < 20) { // Print first 20
        //     printf("  Sample %d: Predicted=%d, Actual=%d\n", i, predicted, actual);
        // }
    }

    double accuracy = (double)correct_predictions / num_test_samples * 100.0;
    printf("Evaluation Complete.\n");
    printf("Accuracy on test set: %.2f%% (%d / %d correct)\n", accuracy, correct_predictions, num_test_samples);

    free_m8(&x_sample);
}


// --- Main Function ---
int main(void) {
    set_mu(LEARNING_RATE_MU);
    // Seed random number generator (important for weight initialization)
    srand(time(NULL));

    // Set the learning rate parameter 'mu' used in network.c
    // This assumes 'mu' is a global or accessible variable in network.c
    // If it's hardcoded, this line won't work and you'd need to modify network.c
    // mu = LEARNING_RATE_MU; // Set the global mu if accessible
    // Note: The provided network.c hardcodes mu=4 inside grad_calc.
    // To change it, you'd need to modify network.c directly or make mu configurable.
    printf("Using learning rate parameter mu = %d\n", LEARNING_RATE_MU);
    printf("Batch Size: %d\n", BATCH_SIZE);


    printf("Initializing Optical Digit Recognition Test...\n");

    // 1. Initialize Data Structures (Allocate max size initially)
    printf("Allocating memory for data...\n");
    Matrix8 X_train_full = init_m8(MAX_TRAIN_SAMPLES, INPUT_SIZE);
    Vector8 Y_train_full = init_v8(MAX_TRAIN_SAMPLES);
    Matrix8 X_test_full = init_m8(MAX_TEST_SAMPLES, INPUT_SIZE);
    Vector8 Y_test_full = init_v8(MAX_TEST_SAMPLES);

    // 2. Load Data
    printf("Loading training data from %s...\n", TRAIN_DATA_FILE);
    int num_train_samples = load_optdigits_data(TRAIN_DATA_FILE, &X_train_full, &Y_train_full, MAX_TRAIN_SAMPLES);
    if (num_train_samples == 0) {
        fprintf(stderr, "Error: No training samples loaded.\n");
        exit(1);
    }

    printf("Loading testing data from %s...\n", TEST_DATA_FILE);
    int num_test_samples = load_optdigits_data(TEST_DATA_FILE, &X_test_full, &Y_test_full, MAX_TEST_SAMPLES);
     if (num_test_samples == 0) {
        fprintf(stderr, "Warning: No test samples loaded. Evaluation will be skipped.\n");
    }

    // 3. Initialize Network
    printf("Initializing network...\n");
    // Network structure: Input (64) -> Hidden (HIDDEN_NEURONS) -> Output (10)
    Network* network = init_network(2, BATCH_SIZE); // 2 layers, batch_size = 1 (SGD)
    // Network* network = init_network(3, BATCH_SIZE); // 2 layers, batch_size = 1 (SGD)

    // Layer 1: Input (INPUT_SIZE) -> Hidden (HIDDEN_NEURONS)
    // Pass next layer pointer, though it seems unused in network_forward/backward
    network->layers[0] = init_layer(BATCH_SIZE, INPUT_SIZE, HIDDEN_NEURONS, LINEAR, network->layers[1]);
    init_weights_xavier_uniform(&network->layers[0]->weights); // Initialize
    printf("Initialized Layer 1 (Hidden): %d inputs, %d neurons\n", INPUT_SIZE, HIDDEN_NEURONS);

    // network->layers[1] = init_layer(BATCH_SIZE, HIDDEN_NEURONS, HIDDEN_NEURONS, RELU, network->layers[2]);
    // printf("Initialized Layer 2 (Hidden): %d inputs, %d neurons\n", HIDDEN_NEURONS, HIDDEN_NEURONS);

    // Layer 2: Hidden (HIDDEN_NEURONS) -> Output (NUM_CLASSES)
    network->layers[1] = init_layer(BATCH_SIZE, HIDDEN_NEURONS, NUM_CLASSES, LINEAR, NULL);
    init_weights_xavier_uniform(&network->layers[1]->weights); // Initialize
    // network->layers[2] = init_layer(BATCH_SIZE, HIDDEN_NEURONS, NUM_CLASSES, LINEAR, NULL);
    printf("Initialized Layer 2 (Output): %d inputs, %d neurons (classes)\n", HIDDEN_NEURONS, NUM_CLASSES);

    // 4. Prepare Per-Sample Data Holders
    // Matrix8 x_sample = init_m8(1, INPUT_SIZE); // Holds one flattened image
    // Vector8 y_sample = init_v8(1);             // Holds one label

    // 5. Training Loop
    printf("\n--- Starting Training for %d Epochs ---\n", NUM_EPOCHS);
    for (uint16_t epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        printf("Epoch %d/%d\n", epoch + 1, NUM_EPOCHS);
        // Optional: Shuffle training data here each epoch for better convergence

        for (int sample_idx = 0; sample_idx < num_train_samples && sample_idx + BATCH_SIZE <= num_train_samples; sample_idx += BATCH_SIZE) {
            // Copy the current sample data into the single-sample holders

            Matrix8 x_sample = m_cpy_range(&X_train_full, sample_idx, sample_idx + BATCH_SIZE - 1);
            Vector8 y_sample = v_cpy_range(&Y_train_full, sample_idx, sample_idx + BATCH_SIZE - 1);
            

            // memcpy(x_sample.matrix[0], X_train_full.matrix[sample_idx], INPUT_SIZE * sizeof(int8_t));
            // x_sample.scale = X_train_full.scale; // Copy scale factor

            // y_sample.vector[0] = Y_train_full.vector[sample_idx];
            // y_sample.scale = Y_train_full.scale;

            // --- Forward Pass ---
            // network_forward returns pointer to last layer's activations
            network_forward(network, &x_sample);
            // DO NOT free out_activations

            // --- Calculate Loss Gradient (for observation/debugging - REDUNDANT) ---
            // network_backward recalculates this internally. Call is optional.
            // Matrix8 loss = loss_gradient(&out_activations, &y_sample); // Uses last layer's activations
            // print_matrix8(&loss, "Loss Grad (Debug)"); // Uncomment to see gradient
            // free_m8(&loss); // Free the matrix returned by loss_gradient

            // --- Backward Pass & Weight Update ---
            // network_backward calculates loss gradient internally, propagates error, updates weights
            Matrix8 err_back_to_input = network_backward(network, &y_sample);
            // Free the error matrix returned by network_backward (if it's newly allocated)
            // Based on network.c, layer_backward calls err_calc which calls get_shift_and_round_bp,
            // which returns a new matrix. So, this needs freeing.
            free_m8(&err_back_to_input);

             // Optional: Print progress
             if ((sample_idx + 1) % 500 == 0) { // Print every 500 samples
                 printf("  Epoch %d: Processed sample %d/%d\n", epoch + 1, sample_idx + 1, num_train_samples);
             }
             
            free_m8(&x_sample);
            free_v8(&y_sample);
        } // End of samples loop

        // Optional: Evaluate on test set after each epoch to see progress
        // evaluate(network, &X_test_full, &Y_test_full, num_test_samples);

        printf("Epoch %d finished.\n", epoch + 1);
        evaluate(network, &X_test_full, &Y_test_full, num_test_samples);

    } // End of epochs loop

    printf("\n--- Training Finished ---\n");

    // 6. Final Evaluation
    evaluate(network, &X_test_full, &Y_test_full, num_test_samples);

    // 7. Print Final Layer State (Optional)
    // print_layer(network->layers[1], "Final Output Layer State");

    // 8. Cleanup
    printf("\nCleaning up resources...\n");
    free_m8(&X_train_full);
    free_v8(&Y_train_full);
    free_m8(&X_test_full);
    free_v8(&Y_test_full);
    free_network(network); // Frees layers and their weights/activations

    printf("Optical Digit Recognition Test Finished.\n");
    return 0;
}
