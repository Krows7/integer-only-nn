#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include "base.h"
#include "dataset_bench_int.h"
#include "network.h"
#include "quantization.h"
// #include "iris-data.h"
#include "iris-data-shuffled.h"
#include <stdlib.h>

#define asm __asm__

#ifdef __NES__
#include <ines.h>

MAPPER_PRG_ROM_KB(32);
MAPPER_PRG_RAM_KB(8);
MAPPER_CHR_ROM_KB(8);
#endif

// --- Dataset Constants ---
#define IRIS_INPUT_SIZE 4       // Sepal Length, Sepal Width, Petal Length, Petal Width
#define IRIS_NUM_CLASSES 3      // Iris-setosa, Iris-versicolor, Iris-virginica
#define IRIS_TOTAL_SAMPLES 150
#define MAX_LINE_LENGTH 255     // Reduced, but still generous
#define TRAIN_SPLIT_RATIO 80   // for training, in %

// Calculate split sizes
#define IRIS_MAX_TRAIN_SAMPLES (int) (IRIS_TOTAL_SAMPLES * TRAIN_SPLIT_RATIO)
#define IRIS_MAX_TEST_SAMPLES (IRIS_TOTAL_SAMPLES - IRIS_MAX_TRAIN_SAMPLES)

// --- Hyperparameters (Adjusted for Iris) ---
#define BATCH_SIZE 4           // Smaller batch size for smaller dataset
#define HIDDEN_NEURONS 8       // Reduced hidden layer size
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

void shuffle_indices(uint8_t *array, size_t n) {
    // if (n > 1) {
    //     for (size_t i = 0; i < n - 1; i++) {
    //       size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
    //       int t = array[j];
    //       array[j] = array[i];
    //       array[i] = t;
    //     }
    // }
}

void split_iris(int8_t*** X_train, Vector8* Y_train, lsize_t* train_count, int8_t*** X_test, Vector8* Y_test, lsize_t* test_count) {
    *train_count = (int16_t) IRIS_SAMPLES * (int16_t) TRAIN_SPLIT_RATIO / (int16_t) 100;
    *test_count = IRIS_SAMPLES - *train_count;

    // TODO TESTS
    // *X_train = (int8_t**) iris_X;
    *X_train = malloc(*train_count * sizeof(int8_t*));
    for (lsize_t i = 0; i < *train_count; ++i) {
        // Cast to (int8_t*) to discard const/volatile for the assignment,
        // assuming read-only access through X_train_float.
        (*X_train)[i] = (int8_t*)iris_X[i]; // If using shuffled indices
        // Or, for a simple sequential split:
        // (*X_train)[i] = (int8_t*)iris_X[i];
    }
    
    // *X_test = iris_X + *train_count;
    *X_test = malloc(*test_count * sizeof(int8_t*));
    for (lsize_t i = 0; i < *test_count; ++i) {
        // Cast to (int8_t*) to discard const/volatile for the assignment,
        // assuming read-only access through X_train_float.
        (*X_test)[i] = (int8_t*)iris_X[i + *train_count]; // If using shuffled indices
        // Or, for a simple sequential split:
        // (*X_train)[i] = (int8_t*)iris_X[i];
    }

    *Y_train = init_v8(*train_count);
    // Y_train->vector = iris_Y;
    free(Y_train->vector);
    Y_train->vector = (int8_t*) iris_Y;

    *Y_test = init_v8(*test_count);
    free(Y_test->vector);
    Y_test->vector = (int8_t*) (iris_Y + *train_count);
    
    return;



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
        // assert_fatal((*X_train)[i], "Memory allocation failed for X_train sample %u", i);
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

#ifdef VBCC
#define MMC3_PRG_RAM_CTRL (*(volatile uint8_t*)0xA001)
#endif

extern void __set_heap_limit(size_t limit);

int main(void) {
    #ifdef __NES__
    __set_heap_limit(1024 * 7);
    #endif
    #ifdef VBCC
    MMC3_PRG_RAM_CTRL = 0x80;
    #endif
    init_pools();
    set_mu(LEARNING_RATE_MU); // Set global learning rate parameter
    // srand(time(NULL)); // Seeding is now done before shuffling in load function

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

    split_iris(&X_train_float, &Y_train_full, &num_train_samples_loaded,
        &X_test_float, &Y_test_full, &num_test_samples_loaded);

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

    train_network(network,
                &X_train_float, &Y_train_full, num_train_samples_loaded,
                &X_test_float, &Y_test_full, num_test_samples_loaded,
                NUM_EPOCHS);
    // cleanup(network, &X_train_float, &Y_train_full, num_train_samples_loaded,
    //                 &X_test_float, &Y_test_full, num_test_samples_loaded);
    free(X_train_float);
    free(X_test_float);
    free_network(network);

    println("Iris Classification Test Finished (C).");

    print_metrics();
    lin_cleanup();
    return 0;
}
