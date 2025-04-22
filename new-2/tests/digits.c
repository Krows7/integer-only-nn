// #include "../ext/dataset_bench.h"
// #include <stdlib.h>
// #include <string.h>
// #include <time.h>
// #include <stdint.h>

// #define LAYERS_SIZE 3

// #define INPUT_SIZE 64
// #define NUM_CLASSES 10
// #define MAX_LINE_LENGTH 512

// // --- Data File Paths ---
// #define TRAIN_DATA_FILE "data/digits/optdigits.tra"
// #define TEST_DATA_FILE "data/digits/optdigits.tes"
// #define MAX_TRAIN_SAMPLES 3823
// #define MAX_TEST_SAMPLES 1797

// // --- Hyperparameters ---
// #define BATCH_SIZE 64
// #define HIDDEN_NEURONS 128
// #define NUM_EPOCHS 10
// #define LEARNING_RATE_MU 4

// // --- Helper Function: Load Optdigits Data into FLOAT arrays ---
// void load_optdigits_float(const char* filename, float*** X_float, Vector8_ext* Y, int max_samples, int* samples_loaded_count) {
//     FILE* file = fopen(filename, "r");
//     if (!file) {
//         fprintf(stderr, "Error: Could not open file %s\n", filename);
//         exit(1);
//     }

//     // Allocate float storage
//     *X_float = malloc(max_samples * sizeof(float*));
//     if (!*X_float) { fprintf(stderr, "Memory allocation failed for X_float pointers\n"); exit(1); }
//     for(int i=0; i<max_samples; ++i) {
//         (*X_float)[i] = malloc(INPUT_SIZE * sizeof(float));
//         if (!(*X_float)[i]) { fprintf(stderr, "Memory allocation failed for X_float sample %d\n", i); exit(1); }
//     }
//     // Y is already allocated (init_v8)

//     char line[MAX_LINE_LENGTH];
//     int sample_count = 0;
//     const char* delim = ",";

//     while (fgets(line, sizeof(line), file) && sample_count < max_samples) {
//         // Basic check for empty or too short lines
//         if (strlen(line) < INPUT_SIZE) continue; // Skip potentially malformed lines

//         char* line_copy = strdup(line); // Use strdup because strtok modifies the string
//         if (!line_copy) { fprintf(stderr, "Memory allocation failed for line_copy\n"); exit(1); }
//         char* token;
//         int feature_count = 0;

//         token = strtok(line_copy, delim);
//         while (token != NULL && feature_count < INPUT_SIZE) {
//             // Check if token is valid number before atoi
//             char *endptr;
//             long val = strtol(token, &endptr, 10);
//             if (endptr == token || *endptr != '\0' || val < 0 || val > 16) {
//                  // Handle error or skip sample, here we print a warning and skip feature
//                  // fprintf(stderr, "Warning: Invalid pixel value '%s' at sample %d, feature %d\n", token, sample_count, feature_count);
//             } else {
//                  (*X_float)[sample_count][feature_count] = (float)val / 16.0f;
//             }
//             feature_count++;
//             token = strtok(NULL, delim);
//         }

//         // Get the label
//         if (token != NULL && feature_count == INPUT_SIZE) {
//              char *endptr;
//              long label_val = strtol(token, &endptr, 10);
//              if (endptr == token || (*endptr != '\0' && *endptr != '\n' && *endptr != '\r') || label_val < 0 || label_val > 9) {
//                  fprintf(stderr, "Warning: Invalid label value '%s' at sample %d\n", token, sample_count);
//                  // Decide how to handle: skip sample or assign default? Skipping for now.
//                  free(line_copy); // Free the duplicated string
//                  continue; // Skip this sample
//              }
//              Y->vector[sample_count] = (int8_t)label_val;
//              sample_count++; // Increment sample count only if valid label found
//         } else if (feature_count != INPUT_SIZE) {
//             fprintf(stderr, "Warning: Incorrect number of features (%d) at sample index %d in %s. Skipping.\n", feature_count, sample_count, filename);
//             // Don't increment sample_count
//         } else {
//              fprintf(stderr, "Warning: Missing label at sample index %d in %s. Skipping.\n", sample_count, filename);
//              // Don't increment sample_count
//         }
//         free(line_copy); // Free the duplicated string
//     }
//     fclose(file);

//     *samples_loaded_count = sample_count;
//     Y->length = sample_count; // Update Y length based on successfully loaded samples
//     Y->scale = 0; // Labels have no scale

//     printf("Loaded %d valid float samples from %s.\n", sample_count, filename);

//     // Note: We don't set X dimensions here, as it's float***
//     // We'll create batch matrices dynamically later.
// }

// // --- Helper Function: Create Quantized Batch from Float Data ---
// // --- Now handles potentially smaller last batch ---
// Matrix8_ext create_quantized_batch(float** full_float_data, int start_idx, int current_batch_size, int total_features) {
//     // 1. Create a temporary float matrix for the batch
//     float** batch_float = malloc(current_batch_size * sizeof(float*));
//     if (!batch_float) { fprintf(stderr, "Memory allocation failed for batch_float pointers\n"); exit(1); }
//     // Point to the rows in the full float data (avoid deep copy)
//     for(int i=0; i<current_batch_size; ++i) {
//         batch_float[i] = full_float_data[start_idx + i];
//     }

//     // 2. Quantize this float batch adaptively
//     // quantize_float_matrix_adaptive needs batch_size and feature_count
//     Matrix8_ext quantized_batch = quantize_float_matrix_adaptive(batch_float, current_batch_size, total_features);

//     // --- DEBUG LOG ---
//     #if DEBUG_LOG_LEVEL >= 1
//     static int input_log_count = 0; // Log only a few times
//     if (input_log_count < 5) {
//         printf("DEBUG [Input]: Quantized Batch (%d samples) Exp = %d\n", current_batch_size, quantized_batch.scale);
//         input_log_count++;
//     }
//     #endif
//     // --- END DEBUG ---

//     // 3. Free the temporary pointer array (not the underlying float data)
//     free(batch_float);

//     return quantized_batch; // Contains int8 data and scale
// }

// void load_train_dataset(float*** X_float, Vector8_ext** Y, int* samples_size) {
//     // float** X_float = NULL;
//     Vector8_ext* Y_full = malloc(sizeof(Vector8_ext));
//     if (!Y_full) { fprintf(stderr, "Memory allocation failed for Y_full\n"); exit(1); }
//     *Y_full = init_v8_ext(MAX_TRAIN_SAMPLES);
//     // Vector8_ext Y_full = init_v8_ext(MAX_TRAIN_SAMPLES);
//     load_optdigits_float(TRAIN_DATA_FILE, X_float, Y_full, MAX_TRAIN_SAMPLES, samples_size);
//     *Y = Y_full;
//     // Matrix8_ext x_batch = create_quantized_batch(X_float, 0, MAX_TRAIN_SAMPLES, INPUT_SIZE);
//     samples_size[0] = MAX_TRAIN_SAMPLES;
//     // Y[0] = Y_full.vector;
//     // X[0] = x_batch.matrix;
// }

// void load_test_dataset(float*** X_float, Vector8_ext** Y, int* samples_size) {
//     // float** X_float = NULL;
//     Vector8_ext* Y_full = malloc(sizeof(Vector8_ext));
//     if (!Y_full) { fprintf(stderr, "Memory allocation failed for Y_full\n"); exit(1); }
//     *Y_full = init_v8_ext(MAX_TRAIN_SAMPLES);
//     // Vector8_ext Y_full = init_v8_ext(MAX_TEST_SAMPLES);
//     load_optdigits_float(TEST_DATA_FILE, X_float, Y_full, MAX_TEST_SAMPLES, samples_size);
//     *Y = Y_full;
//     // Matrix8_ext x_batch = create_quantized_batch(X_float, 0, MAX_TEST_SAMPLES, INPUT_SIZE);
//     samples_size[0] = MAX_TEST_SAMPLES;
//     // Y[0] = Y_full.vector;
//     // X[0] = x_batch.matrix;
// }

// int main(void) {
//     set_mu(LEARNING_RATE_MU);
//     srand(time(NULL));

//     printf("--- C Implementation Configuration ---\n");
//     printf("Using learning rate parameter mu = %d\n", LEARNING_RATE_MU);
//     printf("Batch Size: %d\n", BATCH_SIZE);
//     printf("Target BITWIDTH = %d\n", BITWIDTH); // From linear-math.h
//     printf("Hidden Neurons: %d\n", HIDDEN_NEURONS);
//     printf("Epochs: %d\n", NUM_EPOCHS);
//     printf("-------------------------------------\n");


//     printf("Initializing Optical Digit Recognition Test (C)...\n");


//     float** X_train = NULL;
//     float** X_test = NULL;
//     Vector8_ext* Y_train;
//     Vector8_ext* Y_test;
//     int train_samples = 0;
//     int test_samples = 0;

//     // 1. Allocate Data Structures (Labels only initially)
//     printf("Allocating memory for labels...\n");
//     load_train_dataset(&X_train, &Y_train, &train_samples);

//     printf("Loading testing data (float) from %s...\n", TEST_DATA_FILE);
//     load_test_dataset(&X_test, &Y_test, &test_samples);

//     LayerType kinds[LAYERS_SIZE] = {LINEAR, RELU, LINEAR};
//     uint8_t sizes[LAYERS_SIZE + 1] = {INPUT_SIZE, HIDDEN_NEURONS, HIDDEN_NEURONS, NUM_CLASSES};
    
//     Network* network = create_network(LAYERS_SIZE, kinds, sizes, BATCH_SIZE);
//     // print_network(network);

//     train_network(network, &X_train, Y_train, train_samples, &X_test, Y_test, test_samples, NUM_EPOCHS);

//     return 0;
// }

int main(void) {
    
}