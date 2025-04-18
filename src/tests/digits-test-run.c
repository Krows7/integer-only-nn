#include "../api/network.h" // Includes network logic and implicitly linear-math
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>
#include "../api/weights.h" // Includes init_weights_xavier_uniform

// --- Dataset Constants ---
#define INPUT_SIZE 64
#define NUM_CLASSES 10
#define MAX_LINE_LENGTH 512

// --- Data File Paths ---
#define TRAIN_DATA_FILE "/home/huy/digits/optdigits.tra"
#define TEST_DATA_FILE "/home/huy/digits/optdigits.tes"
#define MAX_TRAIN_SAMPLES 3823
#define MAX_TEST_SAMPLES 1797

// --- Hyperparameters (Match Python where possible) ---
#define BATCH_SIZE 64       // Match ti-digits.py DataLoader
#define HIDDEN_NEURONS 128   // Match ti-digits.py TiLinear
#define NUM_EPOCHS 10        // Match ti-digits.py
#define LEARNING_RATE_MU 4  // Match ti_torch.GRAD_BITWIDTH = 5

// --- Global storage for float data ---
// Store data as floats first for adaptive quantization per batch
float** X_train_float = NULL;
float** X_test_float = NULL;
Vector8 Y_train_full; // Keep labels as int8
Vector8 Y_test_full;
int num_train_samples_loaded = 0;
int num_test_samples_loaded = 0;


// --- Helper Function: Load Optdigits Data into FLOAT arrays ---
void load_optdigits_float(const char* filename, float*** X_float, Vector8* Y, int max_samples, int* samples_loaded_count) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        exit(1);
    }

    // Allocate float storage
    *X_float = malloc(max_samples * sizeof(float*));
    if (!*X_float) { fprintf(stderr, "Memory allocation failed for X_float pointers\n"); exit(1); }
    for(int i=0; i<max_samples; ++i) {
        (*X_float)[i] = malloc(INPUT_SIZE * sizeof(float));
        if (!(*X_float)[i]) { fprintf(stderr, "Memory allocation failed for X_float sample %d\n", i); exit(1); }
    }
    // Y is already allocated (init_v8)

    char line[MAX_LINE_LENGTH];
    int sample_count = 0;
    const char* delim = ",";

    while (fgets(line, sizeof(line), file) && sample_count < max_samples) {
        // Basic check for empty or too short lines
        if (strlen(line) < INPUT_SIZE) continue; // Skip potentially malformed lines

        char* line_copy = strdup(line); // Use strdup because strtok modifies the string
        if (!line_copy) { fprintf(stderr, "Memory allocation failed for line_copy\n"); exit(1); }
        char* token;
        int feature_count = 0;

        token = strtok(line_copy, delim);
        while (token != NULL && feature_count < INPUT_SIZE) {
            // Check if token is valid number before atoi
            char *endptr;
            long val = strtol(token, &endptr, 10);
            if (endptr == token || *endptr != '\0' || val < 0 || val > 16) {
                 // Handle error or skip sample, here we print a warning and skip feature
                 // fprintf(stderr, "Warning: Invalid pixel value '%s' at sample %d, feature %d\n", token, sample_count, feature_count);
            } else {
                 (*X_float)[sample_count][feature_count] = (float)val / 16.0f;
            }
            feature_count++;
            token = strtok(NULL, delim);
        }

        // Get the label
        if (token != NULL && feature_count == INPUT_SIZE) {
             char *endptr;
             long label_val = strtol(token, &endptr, 10);
             if (endptr == token || (*endptr != '\0' && *endptr != '\n' && *endptr != '\r') || label_val < 0 || label_val > 9) {
                 fprintf(stderr, "Warning: Invalid label value '%s' at sample %d\n", token, sample_count);
                 // Decide how to handle: skip sample or assign default? Skipping for now.
                 free(line_copy); // Free the duplicated string
                 continue; // Skip this sample
             }
             Y->vector[sample_count] = (int8_t)label_val;
             sample_count++; // Increment sample count only if valid label found
        } else if (feature_count != INPUT_SIZE) {
            fprintf(stderr, "Warning: Incorrect number of features (%d) at sample index %d in %s. Skipping.\n", feature_count, sample_count, filename);
            // Don't increment sample_count
        } else {
             fprintf(stderr, "Warning: Missing label at sample index %d in %s. Skipping.\n", sample_count, filename);
             // Don't increment sample_count
        }
        free(line_copy); // Free the duplicated string
    }
    fclose(file);

    *samples_loaded_count = sample_count;
    Y->length = sample_count; // Update Y length based on successfully loaded samples
    Y->scale = 0; // Labels have no scale

    printf("Loaded %d valid float samples from %s.\n", sample_count, filename);

    // Note: We don't set X dimensions here, as it's float***
    // We'll create batch matrices dynamically later.
}

// --- Helper Function: Create Quantized Batch from Float Data ---
// --- Now handles potentially smaller last batch ---
Matrix8 create_quantized_batch(float** full_float_data, int start_idx, int current_batch_size, int total_features) {
    // 1. Create a temporary float matrix for the batch
    float** batch_float = malloc(current_batch_size * sizeof(float*));
    if (!batch_float) { fprintf(stderr, "Memory allocation failed for batch_float pointers\n"); exit(1); }
    // Point to the rows in the full float data (avoid deep copy)
    for(int i=0; i<current_batch_size; ++i) {
        batch_float[i] = full_float_data[start_idx + i];
    }

    // 2. Quantize this float batch adaptively
    // quantize_float_matrix_adaptive needs batch_size and feature_count
    Matrix8 quantized_batch = quantize_float_matrix_adaptive(batch_float, current_batch_size, total_features);

    // --- DEBUG LOG ---
    #if DEBUG_LOG_LEVEL >= 1
    static int input_log_count = 0; // Log only a few times
    if (input_log_count < 5) {
        printf("DEBUG [Input]: Quantized Batch (%d samples) Exp = %d\n", current_batch_size, quantized_batch.scale);
        input_log_count++;
    }
    #endif
    // --- END DEBUG ---

    // 3. Free the temporary pointer array (not the underlying float data)
    free(batch_float);

    return quantized_batch; // Contains int8 data and scale
}


// --- Helper Function: Predict Class (Handles Exponents) ---
// --- Now handles potentially smaller last batch ---
Vector8* predict(Network* network, Matrix8* x_batch) {
    // network_forward now returns Matrix8 with data and scale
    // Ensure network_forward can handle x_batch->width != network->batch_size if necessary,
    // or temporarily adjust network's expected batch size. Assuming it handles it for now.
    Matrix8 output_activations = network_forward(network, x_batch);

    Vector8* result = malloc(sizeof(Vector8));
    if (!result) { fprintf(stderr, "Memory allocation failed for prediction result vector\n"); exit(1); }
    result->length = x_batch->width; // Use actual batch size
    result->vector = malloc(result->length * sizeof(int8_t));
     if (!result->vector) { fprintf(stderr, "Memory allocation failed for prediction result data\n"); exit(1); }
    result->scale = 0; // Predictions are class indices

    // Find the index of the maximum activation for each sample in the batch
    for (uint16_t k = 0; k < result->length; ++k) { // Iterate through actual batch size
        // Assuming exponent is same across the class dimension for a given sample:
        int8_t max_activation_int8 = -128;
        int8_t predicted_class = -1;
        // output_activations.height should be NUM_CLASSES
        for (uint8_t i = 0; i < output_activations.height; ++i) {
            // Access should be [sample][class] -> [k][i]
            if (output_activations.matrix[k][i] > max_activation_int8) {
                max_activation_int8 = output_activations.matrix[k][i];
                predicted_class = i;
            }
        }
         result->vector[k] = predicted_class;
    }

    // IMPORTANT: network_forward returns the *actual* last layer activation matrix.
    // We need to free it now as it was copied internally by layer_forward.
    free_m8(&output_activations);

    return result;
}

// --- Helper Function: Evaluate Network (Handles Batches, including partial last batch) ---
void evaluate(Network* network, float** X_float_test, Vector8* Y_test, int num_test_samples) {
    printf("\n--- Evaluating Network (C) ---\n");
    if (num_test_samples == 0) {
        printf("No test samples loaded. Skipping evaluation.\n");
        return;
    }

    int correct_predictions = 0;
    int samples_processed = 0;

    for (int sample_idx = 0; sample_idx < num_test_samples; sample_idx += BATCH_SIZE) {
        // <<< CHANGED: Calculate actual batch size, don't skip partial batch
        int current_batch_size = (sample_idx + BATCH_SIZE <= num_test_samples) ? BATCH_SIZE : (num_test_samples - sample_idx);
        if (current_batch_size <= 0) continue; // Should not happen with proper loop condition

        // Create quantized batch for evaluation
        Matrix8 x_batch = create_quantized_batch(X_float_test, sample_idx, current_batch_size, INPUT_SIZE);

        // Get corresponding labels for the current batch
        // <<< CHANGED: Ensure v_cpy_range handles length correctly
        Vector8 y_batch = v_cpy_range(Y_test, sample_idx, sample_idx + current_batch_size); // end index is exclusive

        // Predict uses the actual size from x_batch
        Vector8* predicted = predict(network, &x_batch);

        // Compare predictions with actual labels
        for (uint16_t k = 0; k < current_batch_size; ++k) { // Use current_batch_size
            if (predicted->vector[k] == y_batch.vector[k]) {
                correct_predictions++;
            }
        }
        samples_processed += current_batch_size;

        // Free memory for this batch
        free_v8(predicted); // Free the result from predict
        free_m8(&x_batch);
        free_v8(&y_batch); // Free the copied labels
    }

    double accuracy = (samples_processed > 0) ? (double)correct_predictions / samples_processed * 100.0 : 0.0;
    printf("Evaluation Complete.\n");
    // <<< CHANGED: Use samples_processed which correctly accounts for partial batches
    printf("Accuracy on test set: %.2f%% (%d / %d correct)\n", accuracy, correct_predictions, samples_processed);
}


// --- Main Function ---
int main(void) {
    set_mu(LEARNING_RATE_MU); // Set global learning rate parameter
    srand(time(NULL));       // Seed random number generator

    printf("--- C Implementation Configuration ---\n");
    printf("Using learning rate parameter mu = %d\n", LEARNING_RATE_MU);
    printf("Batch Size: %d\n", BATCH_SIZE);
    printf("Target BITWIDTH = %d\n", BITWIDTH); // From linear-math.h
    printf("Hidden Neurons: %d\n", HIDDEN_NEURONS);
    printf("Epochs: %d\n", NUM_EPOCHS);
    printf("-------------------------------------\n");


    printf("Initializing Optical Digit Recognition Test (C)...\n");

    // 1. Allocate Data Structures (Labels only initially)
    printf("Allocating memory for labels...\n");
    Y_train_full = init_v8(MAX_TRAIN_SAMPLES);
    Y_test_full = init_v8(MAX_TEST_SAMPLES);

    // 2. Load Data into Float Arrays
    printf("Loading training data (float) from %s...\n", TRAIN_DATA_FILE);
    load_optdigits_float(TRAIN_DATA_FILE, &X_train_float, &Y_train_full, MAX_TRAIN_SAMPLES, &num_train_samples_loaded);
    if (num_train_samples_loaded == 0) {
        fprintf(stderr, "Error: Failed to load any training samples.\n");
        // Free allocated memory before exiting
        free_v8(&Y_train_full);
        free_v8(&Y_test_full);
        // X_train_float might be partially allocated, requires careful freeing or just exit
        exit(1);
    }

    printf("Loading testing data (float) from %s...\n", TEST_DATA_FILE);
    load_optdigits_float(TEST_DATA_FILE, &X_test_float, &Y_test_full, MAX_TEST_SAMPLES, &num_test_samples_loaded);
    if (num_test_samples_loaded == 0) {
        fprintf(stderr, "Warning: Failed to load any testing samples. Evaluation will be skipped.\n");
        // Continue training, but evaluation won't work
    }

    // num_test_samples_loaded = ceil((double) num_test_samples_loaded / 100 * 20);
    // num_train_samples_loaded = MAX_TEST_SAMPLES - num_test_samples_loaded;

    // X_train_float += num_test_samples_loaded;
    // Y_train_full.vector = Y_train_full.vector + num_test_samples_loaded;
    // Y_train_full.length = num_train_samples_loaded;

    // Y_test_full.length = num_test_samples_loaded;

    // 3. Initialize Network (2 layers: Linear -> Linear)
    printf("Initializing network...\n");
    // Network batch size might be used internally, initialize with standard BATCH_SIZE
    // Network* network = init_network(2, BATCH_SIZE);
    Network* network = init_network(3, BATCH_SIZE);

    // Layer 1: Input (64) -> Hidden (64)
    // Assuming init_layer takes (batch_size, input_features, output_neurons, type, next_layer)
    // The batch_size parameter in init_layer might be for pre-allocating intermediate results.
    network->layers[0] = init_layer(BATCH_SIZE, INPUT_SIZE, HIDDEN_NEURONS, LINEAR, NULL); // Set next_layer later if needed
    init_weights_xavier_uniform(&network->layers[0]->weights);
    printf("Initialized Layer 1 (Hidden): %d inputs, %d neurons, Weight Scale: %d\n", INPUT_SIZE, HIDDEN_NEURONS, network->layers[0]->weights.scale);

    network->layers[1] = init_layer(BATCH_SIZE, INPUT_SIZE, HIDDEN_NEURONS, RELU, NULL); // Set next_layer later if needed

    // Layer 2: Hidden (64) -> Output (10)
    network->layers[2] = init_layer(BATCH_SIZE, HIDDEN_NEURONS, NUM_CLASSES, LINEAR, NULL); // Last layer, next is NULL
    init_weights_xavier_uniform(&network->layers[2]->weights);
    // network->layers[1] = init_layer(BATCH_SIZE, HIDDEN_NEURONS, NUM_CLASSES, LINEAR, NULL); // Last layer, next is NULL
    // init_weights_xavier_uniform(&network->layers[1]->weights);
    printf("Initialized Layer 2 (Output): %d inputs, %d neurons, Weight Scale: %d\n", HIDDEN_NEURONS, NUM_CLASSES, network->layers[1]->weights.scale);

    // Link layers (if init_layer didn't handle it) - Assuming network->layers[0] should point to layers[1]
    // This depends on how init_network and init_layer work. If network_forward/backward iterate
    // through network->layers array, linking might not be needed via the 'next' pointer.
    // Let's assume the array iteration is used based on init_network(num_layers, ...).

    // 4. Training Loop
    printf("\n--- Starting Training (C) for %d Epochs ---\n", NUM_EPOCHS);
    for (uint16_t epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        printf("Epoch %d/%d\n", epoch + 1, NUM_EPOCHS);
        // Optional: Shuffle indices here if desired (more complex)
        int samples_processed_epoch = 0;

        for (int sample_idx = 0; sample_idx < num_train_samples_loaded; sample_idx += BATCH_SIZE) {
            // <<< CHANGED: Calculate actual batch size, don't skip partial batch
            int current_batch_size = (sample_idx + BATCH_SIZE <= num_train_samples_loaded) ? BATCH_SIZE : (num_train_samples_loaded - sample_idx);
             if (current_batch_size <= 0) continue; // Safety check

            // <<< REMOVED: Don't skip the last batch
            // if (current_batch_size != BATCH_SIZE) continue;

            // Create quantized batch for training
            // Pass current_batch_size and INPUT_SIZE
            Matrix8 x_batch = create_quantized_batch(X_train_float, sample_idx, current_batch_size, INPUT_SIZE);

            // Get corresponding labels for the current batch
            // Ensure v_cpy_range end index is exclusive: [start, end)
            Vector8 y_batch = v_cpy_range(&Y_train_full, sample_idx, sample_idx + current_batch_size);

            // --- Forward Pass ---
            // network_forward needs to handle the actual batch size in x_batch.width
            Matrix8 out_activations = network_forward(network, &x_batch);
            // We don't need the result directly here, backward pass uses internal state

            // --- Backward Pass & Weight Update ---
            // network_backward needs to handle the actual batch size in y_batch.length
            Matrix8 err_back_to_input = network_backward(network, &y_batch);

            // Free matrices created for the batch
            free_m8(&x_batch);
            free_v8(&y_batch);
            free_m8(&out_activations); // Free the result of forward pass
            free_m8(&err_back_to_input); // Free the result of backward pass

            samples_processed_epoch += current_batch_size;
            // Optional: Print progress (adjust modulo if needed)
            if ((samples_processed_epoch % 512 == 0) || (sample_idx + current_batch_size >= num_train_samples_loaded)) { // Print more often or at end
                 printf("  Epoch %d: Processed %d / %d samples\r", epoch + 1, samples_processed_epoch, num_train_samples_loaded);
                 fflush(stdout); // Ensure progress is shown
            }
        } // End of samples loop
        printf("\nEpoch %d finished. Processed %d samples.\n", epoch + 1, samples_processed_epoch);

        // Evaluate on test set after each epoch
        evaluate(network, X_test_float, &Y_test_full, num_test_samples_loaded);

    } // End of epochs loop

    printf("\n--- Training Finished (C) ---\n");

    // 5. Final Evaluation (already done after last epoch)
    // evaluate(network, X_test_float, &Y_test_full, num_test_samples_loaded);

    // X_train_float -= num_test_samples_loaded;
    // Y_train_full.vector = Y_train_full.vector - num_test_samples_loaded;
    // Y_train_full.length = MAX_TRAIN_SAMPLES;

    // Y_test_full.length = MAX_TRAIN_SAMPLES;

    // 6. Cleanup
    printf("\nCleaning up resources...\n");
    // Free float data
    printf("Freeing training features...\n");
    for(int i=0; i<num_train_samples_loaded; ++i) { // Only free loaded samples
        if (X_train_float[i]) free(X_train_float[i]);
    }
    // Free the outer array only if allocated
    if (X_train_float) free(X_train_float);

    printf("Freeing testing features...\n");
    for(int i=0; i<num_test_samples_loaded; ++i) { // Only free loaded samples
         if (X_test_float[i]) free(X_test_float[i]);
    }
     // Free the outer array only if allocated
    if (X_test_float) free(X_test_float);

    // Free label vectors
    printf("Freeing labels...\n");
    // <<< ADDED: Free label vectors
    free_v8(&Y_train_full);
    free_v8(&Y_test_full);

    // Free network
    printf("Freeing network...\n");
    free_network(network);

    printf("Optical Digit Recognition Test Finished (C).\n");
    return 0;
}