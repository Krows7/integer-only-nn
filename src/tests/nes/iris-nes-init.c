#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "base.h"
#include "dataset_bench_int.h"
#include "quantization.h"
#include "iris-data.h"

// --- Dataset Constants ---
#define IRIS_INPUT_SIZE 4       // Sepal Length, Sepal Width, Petal Length, Petal Width
#define IRIS_NUM_CLASSES 3      // Iris-setosa, Iris-versicolor, Iris-virginica
#define IRIS_TOTAL_SAMPLES 150
#define MAX_LINE_LENGTH 255     // Reduced, but still generous
#define TRAIN_SPLIT_RATIO 80   // for training, in %

#define IRIS_DATA_FILE "data/iris/iris-nes.data"

// Calculate split sizes
#define IRIS_MAX_TRAIN_SAMPLES (int) (IRIS_TOTAL_SAMPLES * TRAIN_SPLIT_RATIO)
#define IRIS_MAX_TEST_SAMPLES (IRIS_TOTAL_SAMPLES - IRIS_MAX_TRAIN_SAMPLES)

// --- Hyperparameters (Adjusted for Iris) ---
#define BATCH_SIZE 16           // Smaller batch size for smaller dataset
#define HIDDEN_NEURONS 64       // Reduced hidden layer size
#define NUM_EPOCHS 10           // May need more epochs for convergence
#define LEARNING_RATE_MU 4      // Keep for now, might need tuning

// --- Global storage for float data ---
// Store data as floats first for adaptive quantization per batch
int8_t** X_train_float = NULL;
int8_t** X_test_float = NULL;
Vector8 Y_train_full; // Keep labels as int8
Vector8 Y_test_full;
lsize_t num_train_samples_loaded = 0;
lsize_t num_test_samples_loaded = 0;


// --- Helper Function: Shuffle indices ---
void shuffle_indices(uint8_t *array, size_t n) {
    if (n > 1) {
        for (size_t i = 0; i < n - 1; i++) {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}


// void load_and_split_iris(const char* filename,
//                                int8_t*** X_train, Vector8* Y_train, lsize_t* train_count,
//                                int8_t*** X_test, Vector8* Y_test, lsize_t* test_count)
// {
//     println("Loading and splitting Iris data (float) from %s...", filename);
//     FILE* file = fopen(filename, "r");
//     assert_fatal(file, "Error: Could not open file %s", filename);

//     int8_t** all_x = malloc(IRIS_TOTAL_SAMPLES * sizeof(int8_t*));
//     int8_t* all_Y_int = malloc(IRIS_TOTAL_SAMPLES * sizeof(int8_t));
//     uint8_t* indices = malloc(IRIS_TOTAL_SAMPLES * sizeof(uint8_t));
//     if (!all_x || !all_Y_int || !indices) {
//         fclose(file);
//         fatal("Memory allocation failed for temporary Iris data storage");
//     }
//     for(int i = 0; i < IRIS_TOTAL_SAMPLES; ++i) {
//         all_x[i] = malloc(IRIS_INPUT_SIZE * sizeof(int8_t));
//         if (!all_x[i]) {
//             for(int j=0; j<i; ++j) free(all_x[j]);
//             free(all_x);
//             free(all_Y_int);
//             free(indices);
//             fclose(file);
//             fatal("Memory allocation failed for temporary Iris sample %d", i);
//         }
//         indices[i] = i;
//     }

//     char line[MAX_LINE_LENGTH];
//     int sample_count = 0;
//     const char* delim = ",";

//     while (fgets(line, sizeof(line), file) && sample_count < IRIS_TOTAL_SAMPLES) {
//         char* line_copy = strdup(line);
//         assert_fatal(line_copy, "Memory allocation failed for line_copy");
//         char* token;
//         int feature_count = 0;

//         token = strtok(line_copy, delim);
//         while (token != NULL && feature_count < IRIS_INPUT_SIZE) {
//             char *endptr;
//             // float val = strtof(token, &endptr);
//             uint8_t val = strtoul(token, &endptr, 10);
//             if (endptr == token || (*endptr != '\0' && *endptr != '\n' && *endptr != '\r' && *endptr != ',')) {
//                 error("Warning: Invalid feature value '%s' at sample %d, feature %d. Skipping sample.", token, sample_count, feature_count);
//                 goto next_line; // Skip rest of the line processing
//             }
//             all_x[sample_count][feature_count] = val;
//             feature_count++;
//             token = strtok(NULL, delim);
//         }

//         if (token != NULL && feature_count == IRIS_INPUT_SIZE) {
//             token[strcspn(token, "\r\n")] = 0;

//             all_Y_int[sample_count] = strtoul(token, NULL, 10);
            
//             sample_count++;
//         } else {
//             error("Warning: Incorrect data format at line corresponding to sample index %d in %s. Skipping.", sample_count, filename);
//         }

//     next_line:
//         free(line_copy);
//     }
//     fclose(file);

//     if (sample_count != IRIS_TOTAL_SAMPLES) {
//         error("Warning: Expected %d samples, but loaded %d valid samples from %s.", IRIS_TOTAL_SAMPLES, sample_count, filename);
//     }
//     log("Loaded %d total valid float samples from %s.", sample_count, filename);

//     shuffle_indices(indices, sample_count);

//     *train_count = (int) (sample_count * TRAIN_SPLIT_RATIO / 100);
//     *test_count = sample_count - *train_count;

//     *X_train = malloc(*train_count * sizeof(int8_t*));
//     *X_test = malloc(*test_count * sizeof(int8_t*));
    
//     assert_fatal(*X_train && X_test, "Memory allocation failed for final train/test X pointers");

//     for (lsize_t i = 0; i < *train_count; ++i) {
//         int original_idx = indices[i];
//         (*X_train)[i] = all_x[original_idx];
//         Y_train->vector[i] = all_Y_int[original_idx];
//     }
//     Y_train->length = *train_count;
//     Y_train->scale = 0;

//     for (lsize_t i = 0; i < *test_count; ++i) {
//         int original_idx = indices[*train_count + i];
//         (*X_test)[i] = all_x[original_idx];
//         Y_test->vector[i] = all_Y_int[original_idx];
//     }
//     Y_test->length = *test_count;
//     Y_test->scale = 0;

//     log("Split data: %d training samples, %d testing samples.", *train_count, *test_count);

//     free(all_x);
//     free(all_Y_int);
//     free(indices);
// }


void split_iris(int8_t*** X_train, Vector8* Y_train, lsize_t* train_count, int8_t*** X_test, Vector8* Y_test, lsize_t* test_count) {
    *train_count = (int16_t) IRIS_SAMPLES * (int16_t) TRAIN_SPLIT_RATIO / (int16_t) 100;
    *test_count = IRIS_SAMPLES - *train_count;

    uint8_t* indices = malloc(IRIS_TOTAL_SAMPLES * sizeof(uint8_t));
    for (uint8_t i = 0; i < IRIS_TOTAL_SAMPLES; ++i) {
        indices[i] = i;
    }
    shuffle_indices(indices, IRIS_TOTAL_SAMPLES);

    *X_train = malloc(*train_count * sizeof(int8_t*));
    assert_fatal(*X_train, "Memory allocation failed for X_train pointers");

    for (lsize_t i = 0; i < *train_count; ++i) {
        // printf("%d\n", i);
        (*X_train)[i] = malloc(IRIS_INPUT_SIZE * sizeof(int8_t));
        assert_fatal((*X_train)[i], "Memory allocation failed for X_train sample %u", i);
        for (lsize_t j = 0; j < IRIS_INPUT_SIZE; ++j) {
            (*X_train)[i][j] = iris_X[indices[i]][j];
        }
    }

    *Y_train = init_v8(*train_count);
    for (lsize_t i = 0; i < *train_count; ++i) {
        (*Y_train).vector[i] = iris_Y[indices[i]];
    }
    Y_train->scale = 0;

    *X_test = malloc(*test_count * sizeof(int8_t*));
    assert_fatal(*X_test, "Memory allocation failed for X_test pointers");
    for (lsize_t i = 0; i < *test_count; ++i) {
        (*X_test)[i] = malloc(IRIS_INPUT_SIZE * sizeof(int8_t));
        assert_fatal((*X_test)[i], "Memory allocation failed for X_test sample %u", i);
        for (lsize_t j = 0; j < IRIS_INPUT_SIZE; ++j) {
            (*X_test)[i][j] = iris_X[indices[i + *train_count]][j];
        }
    }

    *Y_test = init_v8(*test_count);
    for (lsize_t i = 0; i < *test_count; ++i) {
        (*Y_test).vector[i] = iris_Y[indices[i + *train_count]];
    }
    Y_test->scale = 0;

    free(indices);
}

// int main(void) {
//     printf("%d", 1);
// }

#ifdef VBCC
#define MMC3_PRG_RAM_CTRL (*(volatile uint8_t*)0xA001)
#endif

int main(void) {
    #ifdef VBCC
    MMC3_PRG_RAM_CTRL = 0x80;
    #endif
    init_pools();
    set_mu(LEARNING_RATE_MU); // Set global learning rate parameter
    // srand(time(NULL)); // Seeding is now done before shuffling in load function

    print_heap_bounds();

    println("--- C Implementation Configuration (Iris Dataset) ---");
    println("Input Size: %d", IRIS_INPUT_SIZE);
    println("Number of Classes: %d", IRIS_NUM_CLASSES);
    println("Using learning rate parameter mu = %d", LEARNING_RATE_MU);
    println("Batch Size: %d", BATCH_SIZE);
    println("Target BITWIDTH = %d", BITWIDTH);
    println("Hidden Neurons: %d", HIDDEN_NEURONS);
    println("Epochs: %d", NUM_EPOCHS);
    println("Train/Test Split: %d%% / %d%%", TRAIN_SPLIT_RATIO, 100 - TRAIN_SPLIT_RATIO);
    println("----------------------------------------------------");
    println("Initializing Iris Classification Test (C)...");
    println("Allocating memory for labels...");

    // printf("--- C Implementation Configuration (Iris Dataset) ---\n");
    // printf("Input Size: %d\n", IRIS_INPUT_SIZE);
    // printf("Number of Classes: %d\n", IRIS_NUM_CLASSES);
    // printf("Using learning rate parameter mu = %d\n", LEARNING_RATE_MU);
    // printf("Batch Size: %d\n", BATCH_SIZE);
    // printf("Target BITWIDTH = %d\n", BITWIDTH);
    // printf("Hidden Neurons: %d\n", HIDDEN_NEURONS);
    // printf("Epochs: %d\n", NUM_EPOCHS);
    // printf("Train/Test Split: %d%% / %d%%\n", TRAIN_SPLIT_RATIO, 100 - TRAIN_SPLIT_RATIO);
    // printf("----------------------------------------------------\n");
    // printf("Initializing Iris Classification Test (C)...\n");
    // printf("Allocating memory for labels...\n");

    // Y_train_full = init_v8(135);
    // Y_test_full = init_v8(15);

    // load_and_split_iris(IRIS_DATA_FILE,
    //                         &X_train_float, &Y_train_full, &num_train_samples_loaded,
    //                         &X_test_float, &Y_test_full, &num_test_samples_loaded);

    split_iris(&X_train_float, &Y_train_full, &num_train_samples_loaded,
        &X_test_float, &Y_test_full, &num_test_samples_loaded);

    // num_train_samples_loaded = (int16_t) IRIS_SAMPLES * (int16_t) TRAIN_SPLIT_RATIO / (int16_t) 100;
    // num_test_samples_loaded = IRIS_SAMPLES - num_train_samples_loaded;

    // uint8_t* indices = malloc(IRIS_TOTAL_SAMPLES * sizeof(uint8_t));
    // for (uint8_t i = 0; i < IRIS_TOTAL_SAMPLES; ++i) {
    //     indices[i] = i;
    // }
    // shuffle_indices(indices, IRIS_TOTAL_SAMPLES);

    // X_train_float = malloc(num_train_samples_loaded * sizeof(int8_t*));
    // assert_fatal(X_train_float, "Memory allocation failed for X_train pointers");
    // for (lsize_t i = 0; i < 10; ++i) {
    //     // (X_train_float)[i] = malloc(IRIS_INPUT_SIZE * sizeof(int8_t));
    //     (X_train_float)[i] = malloc(4);
    //     // printf("%d ", X_train_float[i]);
    //     // assert_fatal((X_train_float)[i], "Memory allocation failed for X_train sample %u", i);
    //     printf(" ");
    //     for (lsize_t j = 0; j < IRIS_INPUT_SIZE; ++j) {
    //         (X_train_float)[i][j] = iris_X[indices[i]][j];
    //         printf("%d: %d; ", (X_train_float)[i][j], iris_X[indices[i]][j]);
    //     }
    //     printf("\n");
    // }

    // // *Y_train = init_v8(*train_count);
    // // for (lsize_t i = 0; i < *train_count; ++i) {
    // //     (*Y_train).vector[i] = iris_Y[indices[i]];
    // // }
    // // Y_train->scale = 0;

    // // *X_test = malloc(*test_count * sizeof(int8_t*));
    // // assert_fatal(*X_test, "Memory allocation failed for X_test pointers");
    // // for (lsize_t i = 0; i < *test_count; ++i) {
    // //     (*X_test)[i] = malloc(IRIS_INPUT_SIZE * sizeof(int8_t));
    // //     assert_fatal((*X_test)[i], "Memory allocation failed for X_test sample %u", i);
    // //     for (lsize_t j = 0; j < IRIS_INPUT_SIZE; ++j) {
    // //         (*X_test)[i][j] = iris_X[indices[i + *train_count]][j];
    // //     }
    // // }

    // // *Y_test = init_v8(*test_count);
    // // for (lsize_t i = 0; i < *test_count; ++i) {
    // //     (*Y_test).vector[i] = iris_Y[indices[i + *train_count]];
    // // }
    // // Y_test->scale = 0;

    // // free(indices);

    // // printf("3");

    // // // // num_test_samples_loaded = 15;
    // // // // num_train_samples_loaded = 135;

    // // // // X_train_float = malloc(num_train_samples_loaded * sizeof(int8_t*));
    // // // // X_test_float = malloc(num_test_samples_loaded * sizeof(int8_t*));

    // // // // for (lsize_t i = 0; i < num_train_samples_loaded; ++i) {
    // // // //     X_train_float[i] = calloc(IRIS_INPUT_SIZE, sizeof(int8_t));
    // // // //     // memcpy(X_train_float[i], iris_X[i], IRIS_INPUT_SIZE * sizeof(int8_t));
    // // // // }

    // // // // for (lsize_t i = 0; i < num_test_samples_loaded; ++i) {
    // // // //     X_test_float[i] = calloc(IRIS_INPUT_SIZE, sizeof(int8_t));
    // // // //     // memcpy(X_test_float[i], iris_X[i + num_train_samples_loaded], IRIS_INPUT_SIZE * sizeof(int8_t));
    // // // // }

    // // // // memcpy(Y_train_full.vector, iris_Y, num_train_samples_loaded * sizeof(int8_t));
    // // // // Y_train_full.length = num_train_samples_loaded;
    // // // // Y_train_full.scale = 0;

    // // // // memcpy(Y_test_full.vector, iris_Y + num_train_samples_loaded, num_test_samples_loaded * sizeof(int8_t));
    // // // // Y_test_full.length = num_test_samples_loaded;
    // // // // Y_test_full.scale = 0;


    if (num_train_samples_loaded == 0 || num_test_samples_loaded == 0) {
        free_v8(&Y_train_full);
        free_v8(&Y_test_full);
        cleanup(NULL, &X_train_float, &Y_train_full, num_train_samples_loaded,
                    &X_test_float, &Y_test_full, num_test_samples_loaded);
        fatal("Error: Failed to load or split data. Train: %d, Test: %d", num_train_samples_loaded, num_test_samples_loaded);
    }

    Network* network = create_network(3,
                                    (LayerType[]) {LINEAR, RELU, LINEAR},
                                    (lsize_t[]) {IRIS_INPUT_SIZE, HIDDEN_NEURONS, HIDDEN_NEURONS, IRIS_NUM_CLASSES}, // Layer sizes: Input, Hidden, Output
                                    BATCH_SIZE);
    if (!network) {
        cleanup(NULL, &X_train_float, &Y_train_full, num_train_samples_loaded,
                    &X_test_float, &Y_test_full, num_test_samples_loaded);
        fatal("Error: Failed to create network.");
    }

    println("Network created successfully.");
    printf("Network created successfully.\n");
    train_network(network,
                &X_train_float, &Y_train_full, num_train_samples_loaded,
                &X_test_float, &Y_test_full, num_test_samples_loaded,
                NUM_EPOCHS);

    cleanup(network, &X_train_float, &Y_train_full, num_train_samples_loaded,
                    &X_test_float, &Y_test_full, num_test_samples_loaded);

    println("Iris Classification Test Finished (C).");
    printf("Iris Classification Test Finished (C).\n");

    print_metrics();

    lin_cleanup();

    return 0;
}
