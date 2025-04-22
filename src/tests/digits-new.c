#include <string.h>
#include "dataset_bench.h"
#include "float_ops.h"
#include "linear.h"
#include "quantization.h"
#include <stdio.h>

// --- Dataset Constants ---
#define INPUT_SIZE 64
#define NUM_CLASSES 10
#define MAX_LINE_LENGTH 512

// --- Data File Paths ---
#define TRAIN_DATA_FILE "/home/huy/digits/optdigits.tra"
#define TEST_DATA_FILE "/home/huy/digits/optdigits.tes"
#define MAX_TRAIN_SAMPLES 3823
#define MAX_TEST_SAMPLES 1797

// --- Hyperparameters ---
#define BATCH_SIZE 64
#define HIDDEN_NEURONS 128
#define NUM_EPOCHS 10
#define LEARNING_RATE_MU 4

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

int main(void) {
    set_mu(LEARNING_RATE_MU); // Set global learning rate parameter
    // srand(time(NULL));       // Seed random number generator

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

    // 3. Initialize Network
    Network* network = create_network(3, (LayerType[]) {LINEAR, RELU, LINEAR}, (lsize_t[]) {INPUT_SIZE, HIDDEN_NEURONS, HIDDEN_NEURONS, NUM_CLASSES}, BATCH_SIZE);
    
    // 4. Training Loop
    train_network(network, &X_train_float, &Y_train_full, num_train_samples_loaded, &X_test_float, &Y_test_full, num_test_samples_loaded, NUM_EPOCHS);

    // 5. Final Evaluation (already done after last epoch)
    // evaluate(network, X_test_float, &Y_test_full, num_test_samples_loaded);

    // X_train_float -= num_test_samples_loaded;
    // Y_train_full.vector = Y_train_full.vector - num_test_samples_loaded;
    // Y_train_full.length = MAX_TRAIN_SAMPLES;

    // Y_test_full.length = MAX_TRAIN_SAMPLES;

    // 6. Cleanup
    cleanup(network, &X_train_float, &Y_train_full, num_train_samples_loaded, &X_test_float, &Y_test_full, num_test_samples_loaded);

    printf("Optical Digit Recognition Test Finished (C).\n");
    print_metrics();

    // if (*pool) free_m32(*pool);

    lin_cleanup();

    return 0;
}