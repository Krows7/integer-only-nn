#include "../api/network.h" // Includes network logic and implicitly linear-math
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>
#include "../api/weights.h" // Includes init_weights_xavier_uniform

// --- Dataset Constants ---
#define INPUT_SIZE 784      // MNIST images are 28x28 pixels
#define NUM_CLASSES 10      // Digits 0-9
#define MNIST_IMG_MAGIC 0x00000803 // Magic number for image files (big-endian)
#define MNIST_LBL_MAGIC 0x00000801 // Magic number for label files (big-endian)

// --- Data File Paths (Relative to execution directory or adjust as needed) ---
#define TRAIN_IMAGES_FILE "data/MNIST/raw/train-images-idx3-ubyte"
#define TRAIN_LABELS_FILE "data/MNIST/raw/train-labels-idx1-ubyte"
#define TEST_IMAGES_FILE  "data/MNIST/raw/t10k-images-idx3-ubyte"
#define TEST_LABELS_FILE  "data/MNIST/raw/t10k-labels-idx1-ubyte"

#define MAX_TRAIN_SAMPLES 60000
#define MAX_TEST_SAMPLES  10000

// --- Hyperparameters ---
#define BATCH_SIZE 100
#define HIDDEN_NEURONS 256  // Increased hidden layer size for larger input
#define NUM_EPOCHS 5
#define LEARNING_RATE_MU 1  // Corresponds to gradient bit shift (adjust as needed)

// --- Global storage for float data ---
// Store data as floats first for adaptive quantization per batch
float** X_train_float = NULL;
float** X_test_float = NULL;
Vector8 Y_train_full; // Keep labels as int8
Vector8 Y_test_full;
int num_train_samples_loaded = 0;
int num_test_samples_loaded = 0;

// --- Helper Function: Swap Endianness (Big to Little) ---
uint32_t swap_uint32(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

// --- Helper Function: Read MNIST IDX Header ---
// Reads header, performs endian swap, returns number of items.
// Also returns rows and cols for image files (set to 0 for labels).
int read_idx_header(FILE* file, uint32_t expected_magic, uint32_t* num_items, uint32_t* num_rows, uint32_t* num_cols) {
    uint32_t magic_number = 0;
    *num_items = 0;
    *num_rows = 0;
    *num_cols = 0;

    // Read Magic Number
    if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1) return -1; // Read error
    magic_number = swap_uint32(magic_number); // Swap to host byte order
    if (magic_number != expected_magic) {
        fprintf(stderr, "Error: Invalid magic number (Expected 0x%X, Got 0x%X)\n", expected_magic, magic_number);
        return -2; // Magic number mismatch
    }

    // Read Number of Items
    if (fread(num_items, sizeof(uint32_t), 1, file) != 1) return -1;
    *num_items = swap_uint32(*num_items);

    // Read Dimensions (for images)
    if (expected_magic == MNIST_IMG_MAGIC) {
        if (fread(num_rows, sizeof(uint32_t), 1, file) != 1) return -1;
        *num_rows = swap_uint32(*num_rows);
        if (fread(num_cols, sizeof(uint32_t), 1, file) != 1) return -1;
        *num_cols = swap_uint32(*num_cols);
        // Basic check
        if ((*num_rows * *num_cols) != INPUT_SIZE) {
             fprintf(stderr, "Error: Image dimensions (%u x %u = %u) do not match expected INPUT_SIZE (%d)\n",
                     *num_rows, *num_cols, (*num_rows * *num_cols), INPUT_SIZE);
             return -3; // Dimension mismatch
        }
    }
    return 0; // Success
}

// --- Helper Function: Load MNIST Images into FLOAT arrays ---
void load_mnist_images_float(const char* filename, float*** X_float, int max_samples, int* samples_loaded_count) {
    FILE* file = fopen(filename, "rb"); // Open in binary read mode
    if (!file) {
        fprintf(stderr, "Error: Could not open image file %s\n", filename);
        perror("fopen");
        exit(1);
    }

    uint32_t num_items, num_rows, num_cols;
    int header_status = read_idx_header(file, MNIST_IMG_MAGIC, &num_items, &num_rows, &num_cols);

    if (header_status != 0) {
        fprintf(stderr, "Error reading IDX header from %s (Code: %d)\n", filename, header_status);
        fclose(file);
        exit(1);
    }

    if (num_items > max_samples) {
        printf("Warning: File contains %u images, loading only %d.\n", num_items, max_samples);
        num_items = max_samples;
    }

    // Allocate float storage
    *X_float = malloc(num_items * sizeof(float*));
    if (!*X_float) { fprintf(stderr, "Memory allocation failed for X_float pointers\n"); exit(1); }
    for(uint32_t i = 0; i < num_items; ++i) {
        (*X_float)[i] = malloc(INPUT_SIZE * sizeof(float));
        if (!(*X_float)[i]) { fprintf(stderr, "Memory allocation failed for X_float sample %u\n", i); exit(1); }
    }

    // Read image data
    unsigned char pixel;
    for (uint32_t i = 0; i < num_items; ++i) {
        for (int j = 0; j < INPUT_SIZE; ++j) {
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Unexpected end of file while reading image %u, pixel %d from %s\n", i, j, filename);
                // Cleanup partially allocated memory? For simplicity, just exit.
                fclose(file);
                exit(1);
            }
            // Normalize pixel value (0-255) to float (0.0-1.0)
            (*X_float)[i][j] = (float)pixel / 255.0f;
        }
    }

    fclose(file);
    *samples_loaded_count = num_items;
    printf("Loaded %d images (float) from %s.\n", *samples_loaded_count, filename);
}

// --- Helper Function: Load MNIST Labels ---
void load_mnist_labels(const char* filename, Vector8* Y, int expected_samples) {
    FILE* file = fopen(filename, "rb"); // Open in binary read mode
    if (!file) {
        fprintf(stderr, "Error: Could not open label file %s\n", filename);
        perror("fopen");
        exit(1);
    }

    uint32_t num_items, dummy_rows, dummy_cols; // Rows/cols not used for labels
    int header_status = read_idx_header(file, MNIST_LBL_MAGIC, &num_items, &dummy_rows, &dummy_cols);

    if (header_status != 0) {
        fprintf(stderr, "Error reading IDX header from %s (Code: %d)\n", filename, header_status);
        fclose(file);
        exit(1);
    }

    if (num_items != expected_samples) {
        fprintf(stderr, "Warning: Number of labels (%u) in %s does not match number of images (%d). Using %d labels.\n",
                num_items, filename, expected_samples, expected_samples);
        // Adjust num_items if needed, but usually indicates a dataset mismatch.
        // For safety, we'll proceed assuming the image count is correct.
        num_items = expected_samples;
    }

    // Y is already allocated (init_v8)
    if (Y->length < num_items) {
         fprintf(stderr, "Error: Label vector Y is too small (%d) for %u labels.\n", Y->length, num_items);
         fclose(file);
         exit(1);
    }

    // Read label data
    unsigned char label;
    for (uint32_t i = 0; i < num_items; ++i) {
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Unexpected end of file while reading label %u from %s\n", i, filename);
            fclose(file);
            exit(1);
        }
        if (label >= NUM_CLASSES) {
            fprintf(stderr, "Warning: Invalid label value %u at index %u in %s. Clamping to %d.\n", label, i, filename, NUM_CLASSES - 1);
            label = NUM_CLASSES - 1; // Or handle as error
        }
        Y->vector[i] = (int8_t)label; // Store directly as int8_t
    }

    fclose(file);
    Y->length = num_items; // Update actual length
    Y->scale = 0; // Labels have no scale
    printf("Loaded %d labels from %s.\n", Y->length, filename);
}


// --- Helper Function: Create Quantized Batch from Float Data ---
// (Identical to the one in digits-test-run.c - Reusable)
Matrix8 create_quantized_batch(float** full_float_data, int start_idx, int current_batch_size, int total_features) {
    // 1. Create a temporary float matrix for the batch
    float** batch_float = malloc(current_batch_size * sizeof(float*));
    if (!batch_float) { fprintf(stderr, "Memory allocation failed for batch_float pointers\n"); exit(1); }
    // Point to the rows in the full float data (avoid deep copy)
    for(int i=0; i<current_batch_size; ++i) {
        batch_float[i] = full_float_data[start_idx + i];
    }

    // 2. Quantize this float batch adaptively
    Matrix8 quantized_batch = quantize_float_matrix_adaptive(batch_float, current_batch_size, total_features);

    // 3. Free the temporary pointer array (not the underlying float data)
    free(batch_float);

    return quantized_batch; // Contains int8 data and scale
}


// --- Helper Function: Predict Class (Handles Batches) ---
// (Identical to the one in digits-test-run.c - Reusable)
Vector8* predict(Network* network, Matrix8* x_batch) {
    Matrix8 output_activations = network_forward(network, x_batch);

    Vector8* result = malloc(sizeof(Vector8));
    if (!result) { fprintf(stderr, "Memory allocation failed for prediction result vector\n"); exit(1); }
    result->length = x_batch->width; // Use actual batch size
    result->vector = malloc(result->length * sizeof(int8_t));
     if (!result->vector) { fprintf(stderr, "Memory allocation failed for prediction result data\n"); exit(1); }
    result->scale = 0; // Predictions are class indices

    for (uint16_t k = 0; k < result->length; ++k) {
        int8_t max_activation_int8 = -128;
        int8_t predicted_class = -1;
        for (uint8_t i = 0; i < output_activations.height; ++i) { // height should be NUM_CLASSES
            if (output_activations.matrix[k][i] > max_activation_int8) {
                max_activation_int8 = output_activations.matrix[k][i];
                predicted_class = i;
            }
        }
         result->vector[k] = predicted_class;
    }

    // Free the matrix returned by network_forward
    free_m8(&output_activations);

    return result;
}

// --- Helper Function: Evaluate Network (Handles Batches, including partial last batch) ---
// (Almost identical to the one in digits-test-run.c - Reusable, just update print statement)
void evaluate(Network* network, float** X_float_test, Vector8* Y_test, int num_test_samples) {
    printf("\n--- Evaluating Network on MNIST Test Set (C) ---\n");
    if (num_test_samples == 0) {
        printf("No test samples loaded. Skipping evaluation.\n");
        return;
    }

    int correct_predictions = 0;
    int samples_processed = 0;

    for (int sample_idx = 0; sample_idx < num_test_samples; sample_idx += BATCH_SIZE) {
        int current_batch_size = (sample_idx + BATCH_SIZE <= num_test_samples) ? BATCH_SIZE : (num_test_samples - sample_idx);
        if (current_batch_size <= 0) continue;

        Matrix8 x_batch = create_quantized_batch(X_float_test, sample_idx, current_batch_size, INPUT_SIZE);
        Vector8 y_batch = v_cpy_range(Y_test, sample_idx, sample_idx + current_batch_size); // end index is exclusive

        Vector8* predicted = predict(network, &x_batch);

        for (uint16_t k = 0; k < current_batch_size; ++k) {
            if (predicted->vector[k] == y_batch.vector[k]) {
                correct_predictions++;
            }
        }
        samples_processed += current_batch_size;

        free_v8(predicted);
        free_m8(&x_batch);
        free_v8(&y_batch);
    }

    double accuracy = (samples_processed > 0) ? (double)correct_predictions / samples_processed * 100.0 : 0.0;
    printf("Evaluation Complete.\n");
    printf("Accuracy on MNIST test set: %.2f%% (%d / %d correct)\n", accuracy, correct_predictions, samples_processed);
}


// --- Main Function ---
int main(void) {
    set_mu(LEARNING_RATE_MU); // Set global learning rate parameter
    srand(time(NULL));       // Seed random number generator

    printf("--- C MNIST Fixed-Point Network Test ---\n");
    printf("Configuration:\n");
    printf("  Input Size: %d\n", INPUT_SIZE);
    printf("  Hidden Neurons: %d\n", HIDDEN_NEURONS);
    printf("  Output Classes: %d\n", NUM_CLASSES);
    printf("  Batch Size: %d\n", BATCH_SIZE);
    printf("  Epochs: %d\n", NUM_EPOCHS);
    printf("  Learning Rate Mu (Grad Shift): %d\n", LEARNING_RATE_MU);
    printf("  Target BITWIDTH: %d\n", BITWIDTH); // From linear-math.h
    printf("-------------------------------------\n");

    printf("Initializing MNIST Test...\n");

    // 1. Allocate Data Structures (Labels only initially)
    printf("Allocating memory for labels...\n");
    Y_train_full = init_v8(MAX_TRAIN_SAMPLES);
    Y_test_full = init_v8(MAX_TEST_SAMPLES);

    // 2. Load MNIST Data into Float Arrays and Label Vectors
    printf("Loading training data...\n");
    load_mnist_images_float(TRAIN_IMAGES_FILE, &X_train_float, MAX_TRAIN_SAMPLES, &num_train_samples_loaded);
    load_mnist_labels(TRAIN_LABELS_FILE, &Y_train_full, num_train_samples_loaded); // Pass expected count
    if (num_train_samples_loaded == 0 || Y_train_full.length == 0 || num_train_samples_loaded != Y_train_full.length) {
        fprintf(stderr, "Error: Failed to load training data or label mismatch.\n");
        // Basic cleanup before exit
        free_v8(&Y_train_full);
        free_v8(&Y_test_full);
        // Free potentially partially loaded float data (complex, simplified exit here)
        exit(1);
    }

    printf("Loading testing data...\n");
    load_mnist_images_float(TEST_IMAGES_FILE, &X_test_float, MAX_TEST_SAMPLES, &num_test_samples_loaded);
    load_mnist_labels(TEST_LABELS_FILE, &Y_test_full, num_test_samples_loaded); // Pass expected count
    if (num_test_samples_loaded == 0 || Y_test_full.length == 0 || num_test_samples_loaded != Y_test_full.length) {
        fprintf(stderr, "Warning: Failed to load testing data or label mismatch. Evaluation might be skipped or inaccurate.\n");
        // Continue training, but evaluation might fail or give 0%
        num_test_samples_loaded = 0; // Ensure evaluation skips if labels failed
    }


    // 3. Initialize Network (Input -> Linear -> RELU -> Linear -> Output)
    printf("Initializing network...\n");
    // Network requires num_layers and batch_size
    // Network* network = init_network(2, BATCH_SIZE); // 3 computational steps: Linear1, RELU, Linear2
    Network* network = init_network(3, BATCH_SIZE); // 3 computational steps: Linear1, RELU, Linear2

    // Layer 0: Input (784) -> Hidden (Linear)
    network->layers[0] = init_layer(BATCH_SIZE, INPUT_SIZE, HIDDEN_NEURONS, LINEAR, NULL);
    init_weights_xavier_uniform(&network->layers[0]->weights);
    printf("Initialized Layer 0 (Linear): %d inputs, %d neurons, Weight Scale: %d\n", INPUT_SIZE, HIDDEN_NEURONS, network->layers[0]->weights.scale);

    // Layer 1: Activation on Hidden Layer's Output
    // Input size might be conceptually HIDDEN_NEURONS, but init_layer might need original INPUT_SIZE
    // depending on implementation. Let's assume it applies activation element-wise based on previous layer's output size.
    // The input/output size for RELU layer in init_layer might be ignored or used for internal buffer allocation.
    // Check your init_layer implementation detail if issues arise.
    network->layers[1] = init_layer(BATCH_SIZE, HIDDEN_NEURONS, HIDDEN_NEURONS, RELU, NULL);
    printf("Initialized Layer 1 (RELU Activation)\n");


    // Layer 2: Hidden -> Output (Linear)
    // network->layers[1] = init_layer(BATCH_SIZE, HIDDEN_NEURONS, NUM_CLASSES, LINEAR, NULL);
    // init_weights_xavier_uniform(&network->layers[1]->weights);
    // printf("Initialized Layer 2 (Linear): %d inputs, %d neurons, Weight Scale: %d\n", HIDDEN_NEURONS, NUM_CLASSES, network->layers[1]->weights.scale);
    network->layers[2] = init_layer(BATCH_SIZE, HIDDEN_NEURONS, NUM_CLASSES, LINEAR, NULL);
    init_weights_xavier_uniform(&network->layers[2]->weights);
    printf("Initialized Layer 2 (Linear): %d inputs, %d neurons, Weight Scale: %d\n", HIDDEN_NEURONS, NUM_CLASSES, network->layers[2]->weights.scale);

    // 4. Training Loop
    printf("\n--- Starting Training (C) for %d Epochs ---\n", NUM_EPOCHS);
    for (uint16_t epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        printf("Epoch %d/%d\n", epoch + 1, NUM_EPOCHS);
        int samples_processed_epoch = 0;

        // Optional: Add shuffling of training indices here for better training

        for (int sample_idx = 0; sample_idx < num_train_samples_loaded; sample_idx += BATCH_SIZE) {
            int current_batch_size = (sample_idx + BATCH_SIZE <= num_train_samples_loaded) ? BATCH_SIZE : (num_train_samples_loaded - sample_idx);
             if (current_batch_size <= 0) continue;

            // Create quantized batch for training
            Matrix8 x_batch = create_quantized_batch(X_train_float, sample_idx, current_batch_size, INPUT_SIZE);

            // Get corresponding labels for the current batch
            Vector8 y_batch = v_cpy_range(&Y_train_full, sample_idx, sample_idx + current_batch_size);

            // --- Forward Pass ---
            Matrix8 out_activations = network_forward(network, &x_batch);

            // --- Backward Pass & Weight Update ---
            Matrix8 err_back_to_input = network_backward(network, &y_batch);

            // Free matrices created for the batch
            free_m8(&x_batch);
            free_v8(&y_batch);
            free_m8(&out_activations); // Free the result of forward pass
            free_m8(&err_back_to_input); // Free the result of backward pass

            samples_processed_epoch += current_batch_size;
            // Print progress
            if ((samples_processed_epoch % (BATCH_SIZE * 20) == 0) || (sample_idx + current_batch_size >= num_train_samples_loaded)) {
                 printf("  Epoch %d: Processed %d / %d samples\r", epoch + 1, samples_processed_epoch, num_train_samples_loaded);
                 fflush(stdout);
            }
        } // End of samples loop
        printf("\nEpoch %d finished. Processed %d samples.\n", epoch + 1, samples_processed_epoch);

        // Evaluate on test set after each epoch
        if (num_test_samples_loaded > 0) {
            evaluate(network, X_test_float, &Y_test_full, num_test_samples_loaded);
        } else {
            printf("Skipping evaluation as test data failed to load.\n");
        }

    } // End of epochs loop

    printf("\n--- Training Finished (C) ---\n");

    // 5. Final Evaluation (already done after last epoch)
    // if (num_test_samples_loaded > 0) {
    //     evaluate(network, X_test_float, &Y_test_full, num_test_samples_loaded);
    // }

    // 6. Cleanup
    printf("\nCleaning up resources...\n");
    // Free float data
    printf("Freeing training features...\n");
    if (X_train_float) {
        for(int i=0; i<num_train_samples_loaded; ++i) {
            if (X_train_float[i]) free(X_train_float[i]);
        }
        free(X_train_float);
    }

    printf("Freeing testing features...\n");
    if (X_test_float) {
        for(int i=0; i<num_test_samples_loaded; ++i) {
             if (X_test_float[i]) free(X_test_float[i]);
        }
        free(X_test_float);
    }

    // Free label vectors
    printf("Freeing labels...\n");
    free_v8(&Y_train_full);
    free_v8(&Y_test_full);

    // Free network
    printf("Freeing network...\n");
    free_network(network);

    printf("MNIST Fixed-Point Network Test Finished (C).\n");
    return 0;
}