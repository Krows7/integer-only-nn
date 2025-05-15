#include <stdint.h>
#include <stdlib.h> // Added for malloc, free, exit, srand, rand
#include <string.h>
#include <time.h>   // Added for srand
#include "dataset_bench.h"
#include "float_ops.h"
#include "linear.h"
#include "quantization.h"

// --- Dataset Constants ---
#define IRIS_INPUT_SIZE 4       // Sepal Length, Sepal Width, Petal Length, Petal Width
#define IRIS_NUM_CLASSES 3      // Iris-setosa, Iris-versicolor, Iris-virginica
#define IRIS_TOTAL_SAMPLES 150
#define MAX_LINE_LENGTH 256     // Reduced, but still generous
#define TRAIN_SPLIT_RATIO 0.9   // 80% for training

// --- Data File Paths ---
// IMPORTANT: Update this path to the location of your iris.data file
#define IRIS_DATA_FILE "data/iris/iris.data" // <--- CHANGE THIS

// Calculate split sizes
#define IRIS_MAX_TRAIN_SAMPLES (int)(IRIS_TOTAL_SAMPLES * TRAIN_SPLIT_RATIO)
#define IRIS_MAX_TEST_SAMPLES (IRIS_TOTAL_SAMPLES - IRIS_MAX_TRAIN_SAMPLES)

// --- Hyperparameters (Adjusted for Iris) ---
#define BATCH_SIZE 16           // Smaller batch size for smaller dataset
#define HIDDEN_NEURONS 128       // Reduced hidden layer size
#define NUM_EPOCHS 10           // May need more epochs for convergence
#define LEARNING_RATE_MU 4      // Keep for now, might need tuning

// --- Global storage for float data ---
// Store data as floats first for adaptive quantization per batch
float** X_train_float = NULL;
float** X_test_float = NULL;
Vector8 Y_train_full; // Keep labels as int8
Vector8 Y_test_full;
int num_train_samples_loaded = 0;
int num_test_samples_loaded = 0;


// --- Helper Function: Shuffle indices ---
void shuffle_indices(int *array, size_t n) {
    if (n > 1) {
        for (size_t i = 0; i < n - 1; i++) {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}


// --- Helper Function: Load Iris Data into FLOAT arrays and split ---
void load_and_split_iris(const char* filename,
                               float*** X_train, Vector8* Y_train, int* train_count,
                               float*** X_test, Vector8* Y_test, int* test_count)
{
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        exit(1);
    }

    // --- Temporary storage for the full dataset ---
    float** all_X_float = malloc(IRIS_TOTAL_SAMPLES * sizeof(float*));
    int8_t* all_Y_int = malloc(IRIS_TOTAL_SAMPLES * sizeof(int8_t));
    int* indices = malloc(IRIS_TOTAL_SAMPLES * sizeof(int));
    if (!all_X_float || !all_Y_int || !indices) {
        fprintf(stderr, "Memory allocation failed for temporary Iris data storage\n");
        fclose(file);
        exit(1);
    }
    for(int i=0; i<IRIS_TOTAL_SAMPLES; ++i) {
        all_X_float[i] = malloc(IRIS_INPUT_SIZE * sizeof(float));
        if (!all_X_float[i]) {
            fprintf(stderr, "Memory allocation failed for temporary Iris sample %d\n", i);
            // Basic cleanup before exit
            for(int j=0; j<i; ++j) free(all_X_float[j]);
            free(all_X_float);
            free(all_Y_int);
            free(indices);
            fclose(file);
            exit(1);
        }
        indices[i] = i; // Initialize indices for shuffling
    }

    char line[MAX_LINE_LENGTH];
    int sample_count = 0;
    const char* delim = ",";

    while (fgets(line, sizeof(line), file) && sample_count < IRIS_TOTAL_SAMPLES) {
        // Basic check for empty lines or lines too short
        if (strlen(line) < IRIS_INPUT_SIZE * 2) continue; // Rough check

        char* line_copy = strdup(line); // Use strdup because strtok modifies the string
        if (!line_copy) { fprintf(stderr, "Memory allocation failed for line_copy\n"); exit(1); }
        char* token;
        int feature_count = 0;

        // Parse features
        token = strtok(line_copy, delim);
        while (token != NULL && feature_count < IRIS_INPUT_SIZE) {
            char *endptr;
            float val = strtof(token, &endptr); // Use strtof for float parsing
            if (endptr == token || (*endptr != '\0' && *endptr != '\n' && *endptr != '\r' && *endptr != ',')) {
                 fprintf(stderr, "Warning: Invalid feature value '%s' at sample %d, feature %d. Skipping sample.\n", token, sample_count, feature_count);
                 goto next_line; // Skip rest of the line processing
            }
            all_X_float[sample_count][feature_count] = val; // Store directly as float
            feature_count++;
            token = strtok(NULL, delim);
        }

        // Parse the label
        if (token != NULL && feature_count == IRIS_INPUT_SIZE) {
             // Remove trailing newline/carriage return if present
             token[strcspn(token, "\r\n")] = 0;

             if (strcmp(token, "Iris-setosa") == 0) {
                 all_Y_int[sample_count] = 0;
             } else if (strcmp(token, "Iris-versicolor") == 0) {
                 all_Y_int[sample_count] = 1;
             } else if (strcmp(token, "Iris-virginica") == 0) {
                 all_Y_int[sample_count] = 2;
             } else {
                 fprintf(stderr, "Warning: Unknown label value '%s' at sample %d. Skipping sample.\n", token, sample_count);
                 goto next_line; // Skip rest of the line processing
             }
             sample_count++; // Increment sample count only if valid sample loaded
        } else {
            fprintf(stderr, "Warning: Incorrect data format at line corresponding to sample index %d in %s. Skipping.\n", sample_count, filename);
            // Don't increment sample_count
        }

    next_line:
        free(line_copy); // Free the duplicated string
    }
    fclose(file);

    if (sample_count != IRIS_TOTAL_SAMPLES) {
         fprintf(stderr, "Warning: Expected %d samples, but loaded %d valid samples from %s.\n", IRIS_TOTAL_SAMPLES, sample_count, filename);
         // Adjust total samples if needed, though this indicates a file issue
         // For simplicity, we'll proceed but this might cause issues later if counts mismatch defines
    }
    printf("Loaded %d total valid float samples from %s.\n", sample_count, filename);


    // --- Shuffle and Split ---
    // srand(time(NULL)); // Seed random number generator for shuffling
    shuffle_indices(indices, sample_count);

    *train_count = (int)(sample_count * TRAIN_SPLIT_RATIO);
    *test_count = sample_count - *train_count;

    // Allocate final storage
    *X_train = malloc(*train_count * sizeof(float*));
    *X_test = malloc(*test_count * sizeof(float*));
    // Y_train and Y_test are assumed to be pre-allocated by init_v8 in main
    if (!*X_train || !*X_test) {
        fprintf(stderr, "Memory allocation failed for final train/test X pointers\n");
        // Proper cleanup is more complex here, exiting for simplicity
        exit(1);
    }

    // Copy data based on shuffled indices
    for (int i = 0; i < *train_count; ++i) {
        int original_idx = indices[i];
        (*X_train)[i] = all_X_float[original_idx]; // Transfer ownership of the malloc'd row
        Y_train->vector[i] = all_Y_int[original_idx];
    }
    Y_train->length = *train_count;
    Y_train->scale = 0; // Labels have no scale

    for (int i = 0; i < *test_count; ++i) {
        int original_idx = indices[*train_count + i];
        (*X_test)[i] = all_X_float[original_idx]; // Transfer ownership of the malloc'd row
        Y_test->vector[i] = all_Y_int[original_idx];
    }
    Y_test->length = *test_count;
    Y_test->scale = 0; // Labels have no scale

    printf("Split data: %d training samples, %d testing samples.\n", *train_count, *test_count);

    // --- Cleanup temporary storage ---
    // We transferred ownership of the inner float arrays, so only free the outer pointers and Y array
    free(all_X_float); // Free the array of pointers, not the pointed-to data
    free(all_Y_int);
    free(indices);
}


// --- Helper Function: Create Quantized Batch from Float Data ---
// --- (Identical to the original, should work fine) ---
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
        printf("DEBUG [Input]: Quantized Batch (%d samples, %d features) Exp = %d\n", current_batch_size, total_features, quantized_batch.scale);
        input_log_count++;
    }
    #endif
    // --- END DEBUG ---

    // 3. Free the temporary pointer array (not the underlying float data)
    free(batch_float);

    return quantized_batch; // Contains int8 data and scale
}


int main(void) {
    init_pools();
    set_mu(LEARNING_RATE_MU); // Set global learning rate parameter
    // srand(time(NULL)); // Seeding is now done before shuffling in load function

    printf("--- C Implementation Configuration (Iris Dataset) ---\n");
    printf("Input Size: %d\n", IRIS_INPUT_SIZE);
    printf("Number of Classes: %d\n", IRIS_NUM_CLASSES);
    printf("Using learning rate parameter mu = %d\n", LEARNING_RATE_MU);
    printf("Batch Size: %d\n", BATCH_SIZE);
    printf("Target BITWIDTH = %d\n", BITWIDTH); // From linear-math.h or similar
    printf("Hidden Neurons: %d\n", HIDDEN_NEURONS);
    printf("Epochs: %d\n", NUM_EPOCHS);
    printf("Train/Test Split: %.2f / %.2f\n", TRAIN_SPLIT_RATIO, 1.0 - TRAIN_SPLIT_RATIO);
    printf("----------------------------------------------------\n");


    printf("Initializing Iris Classification Test (C)...\n");

    // 1. Allocate Data Structures (Labels only initially)
    printf("Allocating memory for labels...\n");
    // Allocate slightly more just in case rounding differs, length will be set precisely
    Y_train_full = init_v8(IRIS_TOTAL_SAMPLES);
    Y_test_full = init_v8(IRIS_TOTAL_SAMPLES);


    // 2. Load Data into Float Arrays and Split
    printf("Loading and splitting Iris data (float) from %s...\n", IRIS_DATA_FILE);
    load_and_split_iris(IRIS_DATA_FILE,
                              &X_train_float, &Y_train_full, &num_train_samples_loaded,
                              &X_test_float, &Y_test_full, &num_test_samples_loaded);

    if (num_train_samples_loaded == 0 || num_test_samples_loaded == 0) {
        fprintf(stderr, "Error: Failed to load or split data. Train: %d, Test: %d\n",
                num_train_samples_loaded, num_test_samples_loaded);
        // Basic cleanup before exiting
        free_v8(&Y_train_full);
        free_v8(&Y_test_full);
        // X arrays might be partially allocated or NULL, cleanup_iris handles NULL
        cleanup(NULL, &X_train_float, &Y_train_full, num_train_samples_loaded,
                     &X_test_float, &Y_test_full, num_test_samples_loaded);
        exit(1);
    }

    // 3. Initialize Network
    // Network structure: Linear -> ReLU -> Linear (Output)
    // Sizes: Input -> Hidden -> Output
    Network* network = create_network(3, // Number of layers (Linear, ReLU, Linear)
                                     (LayerType[]) {LINEAR, RELU, LINEAR},
                                     (lsize_t[]) {IRIS_INPUT_SIZE, HIDDEN_NEURONS, HIDDEN_NEURONS, IRIS_NUM_CLASSES}, // Layer sizes: Input, Hidden, Output
                                     BATCH_SIZE);
    if (!network) {
         fprintf(stderr, "Error: Failed to create network.\n");
         cleanup(NULL, &X_train_float, &Y_train_full, num_train_samples_loaded,
                      &X_test_float, &Y_test_full, num_test_samples_loaded);
         exit(1);
    }
    printf("Network created successfully.\n");

    // 4. Training Loop
    // Assuming train_network takes float*** for X data and handles batch creation internally
    // using a function like create_quantized_batch provided earlier.
    train_network(network,
                  &X_train_float, &Y_train_full, num_train_samples_loaded,
                  &X_test_float, &Y_test_full, num_test_samples_loaded,
                  NUM_EPOCHS);

    // 5. Final Evaluation (often done within the last epoch of train_network)
    // If not, you might call an evaluate function here:
    // evaluate(network, X_test_float, &Y_test_full, num_test_samples_loaded);

    // 6. Cleanup
    cleanup(network, &X_train_float, &Y_train_full, num_train_samples_loaded,
                     &X_test_float, &Y_test_full, num_test_samples_loaded);

    printf("Iris Classification Test Finished (C).\n");
    print_metrics(); // Assuming this exists and prints relevant stats

    lin_cleanup(); // Assuming this is a general cleanup for the linear algebra library

    return 0;
}
