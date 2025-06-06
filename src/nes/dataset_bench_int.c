#include "dataset_bench_int.h"
#include "base.h"
#include "linear.h"
#include "network.h"
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// integer square-root (any standard 32-bit routine will do)
uint32_t isqrt32(uint32_t x) {
    uint32_t res = 0;
    uint32_t bit = 1UL << 30;
    while (bit > x) bit >>= 2;
    for (; bit; bit >>= 2) {
        if (x >= res + bit) {
            x -= res + bit;
            res = (res >> 1) + bit;
        } else {
            res >>= 1;
        }
    }
    return res;
}

// w ~ U(-l, l), l = sqrt(6/(in+out))
void init_xavier(Matrix8* weights) {
    uint32_t sum = weights->width + weights->height;
    uint32_t ratio = (6UL << 16) / sum;
    // uint32_t ratio = 32768;
    uint32_t limit = isqrt32(ratio);
    if (limit > 255) limit = 255;
    weights->scale = -8;

    for (lsize_t i = 0; i < weights->width; i++) {
        for (lsize_t j = 0; j < weights->height; j++) {
            int16_t r16 = rand16();
            // println("%d", r16);
            int16_t v = (int16_t) ((r16 * limit) >> 15);
            // println("%d %d %lu %lu %lu", r16, v, sum, ratio, limit);
            weights->matrix[i][j] = (int8_t)v;
        }
    }
}

void init_weights(Matrix8 *weights) {
    init_xavier(weights);
}

__bank(1) Network* create_network(lsize_t layers_size, const LayerType* kinds, const lsize_t* sizes, lsize_t batch_size) {
    nes_srand_32(1);
    println("Initializing network...");
    Network* network = init_network(layers_size, batch_size);
    println("Initializing layers...");
    for (lsize_t i = 0; i < layers_size; ++i) {
        network->layers[i] = init_layer(batch_size, sizes[i], sizes[i + 1], kinds[i]);
        if (kinds[i] == LINEAR) {
            init_weights(&network->layers[i]->weights);
            // print_matrix8(&network->layers[i]->weights, "Weights");
        }
    }
    print_network(network);
    return network;
}

Network* create_raw_network(lsize_t layers_size, const LayerType* kinds, const lsize_t* sizes, lsize_t batch_size) {
    // nes_srand_32(1);
    // println("Initializing network...");
    Network* network = init_network(layers_size, batch_size);
    // println("Initializing layers...");
    for (lsize_t i = 0; i < layers_size; ++i) {
        network->layers[i] = init_layer(batch_size, sizes[i], sizes[i + 1], kinds[i]);
    }
    // print_network(network);
    return network;
}

__bank(1) Vector8* predict(const Network* network, const Matrix8* x_batch) {
    Matrix8 output_activations = network_forward(network, x_batch);

    Vector8* result = (Vector8*) malloc(sizeof(Vector8));
    assert_fatal(result, "Memory allocation failed for prediction result vector");
    result->length = x_batch->width;
    result->vector = (int8_t*) malloc(result->length * sizeof(int8_t));
    assert_fatal(result->vector, "Memory allocation failed for prediction result data");
    result->scale = 0;

    for (lsize_t k = 0; k < result->length; ++k) {
        int8_t max_activation_int8 = -128;
        int8_t predicted_class = -1;
        for (uint8_t i = 0; i < output_activations.height; ++i) {
            if (output_activations.matrix[k][i] > max_activation_int8) {
                max_activation_int8 = output_activations.matrix[k][i];
                predicted_class = i;
            }
        }
        result->vector[k] = predicted_class;
    }
    free_m8(&output_activations);
    return result;
}

__bank(1) void evaluate(const Network* network, int8_t*** X_test, const Vector8* Y_test, lsize_t num_test_samples) {
    lsize_t batch_size = network->batch_size;
    lsize_t input_size = network->layers[0]->weights.height;
    println("\n--- Evaluating Network (C) ---");
    if (num_test_samples == 0) {
        println("No test samples loaded. Skipping evaluation.");
        return;
    }

    uint8_t correct_predictions = 0;
    uint8_t samples_processed = 0;

    Matrix8 x_batch;
    Vector8 y_batch;

    for (lsize_t sample_idx = 0; sample_idx < num_test_samples; sample_idx += batch_size) {
        lsize_t current_batch_size = (sample_idx + batch_size <= num_test_samples) ? batch_size : (num_test_samples - sample_idx);

        load_batch(X_test, Y_test, &x_batch, &y_batch, sample_idx, current_batch_size, input_size);

        Vector8* predicted = predict(network, &x_batch);

        for (lsize_t k = 0; k < current_batch_size; ++k) {
            if (predicted->vector[k] == y_batch.vector[k]) {
                correct_predictions++;
            }
        }
        samples_processed += current_batch_size;

        free_v8(predicted);
        free(predicted);
        free_m8(&x_batch);
        free_v8(&y_batch);
    }

    #ifndef NO_PRINT
    uint8_t accuracy = (uint8_t) ((uint16_t) correct_predictions * 100 / samples_processed);
    println("Evaluation Complete.");
    println("Accuracy on test set: ~%d%% (%d / %d correct)", accuracy, correct_predictions, samples_processed);
    #endif
}

__bank(1) void evaluate_full(const Network* network, int8_t*** X_test, const Vector8* Y_test, lsize_t num_test_samples, int8_t*** X_train, const Vector8* Y_train, lsize_t num_train_samples) {
    lsize_t batch_size = network->batch_size;
    lsize_t input_size = network->layers[0]->weights.height;
    println("\n--- Evaluating Network (C) ---");
    if (num_test_samples == 0) {
        println("No test samples loaded. Skipping evaluation.");
        return;
    }

    uint8_t correct_predictions = 0;
    uint8_t samples_processed = 0;

    Matrix8 x_batch;
    Vector8 y_batch;

    for (lsize_t sample_idx = 0; sample_idx < num_test_samples; sample_idx += batch_size) {
        lsize_t current_batch_size = (sample_idx + batch_size <= num_test_samples) ? batch_size : (num_test_samples - sample_idx);

        load_batch(X_test, Y_test, &x_batch, &y_batch, sample_idx, current_batch_size, input_size);

        Vector8* predicted = predict(network, &x_batch);

        for (lsize_t k = 0; k < current_batch_size; ++k) {
            if (predicted->vector[k] == y_batch.vector[k]) {
                correct_predictions++;
            }
        }
        samples_processed += current_batch_size;

        free_v8(predicted);
        free(predicted);
        free_m8(&x_batch);
        free_v8(&y_batch);
    }

    #ifndef NO_PRINT
    uint16_t accuracy = (uint16_t) ((uint32_t) correct_predictions * 10000 / samples_processed);
    println("Evaluation Complete.");
    // println("Accuracy on test set: ~%d%% (%d / %d correct)", accuracy, correct_predictions, samples_processed);
    println("Accuracy on test set: %d.%d%% (%d / %d correct)", accuracy / 100, accuracy % 100, correct_predictions, samples_processed);
    #endif

    correct_predictions = 0;
    samples_processed = 0;

    for (lsize_t sample_idx = 0; sample_idx < num_train_samples; sample_idx += batch_size) {
        lsize_t current_batch_size = (sample_idx + batch_size <= num_train_samples) ? batch_size : (num_train_samples - sample_idx);

        load_batch(X_train, Y_train, &x_batch, &y_batch, sample_idx, current_batch_size, input_size);

        Vector8* predicted = predict(network, &x_batch);

        for (lsize_t k = 0; k < current_batch_size; ++k) {
            if (predicted->vector[k] == y_batch.vector[k]) {
                correct_predictions++;
            }
        }
        samples_processed += current_batch_size;

        free_v8(predicted);
        free(predicted);
        free_m8(&x_batch);
        free_v8(&y_batch);
    }

    #ifndef NO_PRINT
    accuracy = (uint16_t) ((uint32_t) correct_predictions * 10000 / samples_processed);
    println("Accuracy on train set: %d.%d%% (%d / %d correct)", accuracy / 100, accuracy % 100, correct_predictions, samples_processed);
    #endif
}

void print_loss(const Matrix8* loss) {
    int32_t sum = 0;
    for (lsize_t i = 0; i < loss->width; ++i) {
        for (lsize_t j = 0; j < loss->height; ++j) {
            sum += loss->matrix[j][i];
        }
    }
    println("Loss [Sum: " FMT_32 "; Mean: " FMT_32 " (Scale = " FMT_8 ")]", sum, sum / (loss->width * loss->height), loss->scale);
}

const char* num = "%d\n";

__bank(2) void train_network(const Network* network, int8_t*** X_train, Vector8* Y_train, lsize_t train_samples_size, int8_t*** X_test, Vector8* Y_test, lsize_t test_samples_size, uint32_t epochs) {
    nes_srand_32(1);
    lsize_t batch_size = network->batch_size;
    lsize_t input_size = network->layers[0]->weights.height;
    println("\n--- Starting Training (C) for " FMT_u32 " Epochs ---", epochs);
    Matrix8 x_batch;
    Vector8 y_batch;
    for (uint32_t epoch = 0; epoch < epochs; ++epoch) {
        println("\nEpoch " FMT_u32 "/" FMT_u32, epoch + 1, epochs);
        lsize_t samples_processed_epoch = 0;
        // int32_t loss_sum = 0;

        for (lsize_t sample_idx = 0; sample_idx < train_samples_size; sample_idx += batch_size) {
            lsize_t current_batch_size = (sample_idx + batch_size <= train_samples_size) ? batch_size : (train_samples_size - sample_idx);
            if (current_batch_size <= 0) continue;

            load_batch(X_train, Y_train, &x_batch, &y_batch, sample_idx, current_batch_size, input_size);

            // print_matrix8(&x_batch, "x_batch");
            // print_vector8(&y_batch, "y_batch");
            

            Matrix8 out_activations = network_forward(network, &x_batch);

            // print_matrix8(&out_activations, "Forward Out");

            // Matrix8 loss = loss_gradient(&out_activations, &y_batch);

            // loss_sum += m8_sum(&loss);

            // print_loss(&loss);

            Matrix8 err_back_to_input = network_backward(network, &y_batch);

            free_m8(&x_batch);
            free_v8(&y_batch);
            free_m8(&out_activations);
            free_m8(&err_back_to_input);

            #ifndef NO_PRINT
            samples_processed_epoch += current_batch_size;
            if ((samples_processed_epoch % 512 == 0) || (sample_idx + current_batch_size >= train_samples_size)) {
                // printf("Epoch " FMT_u32 ": Processed " FMT_LSIZE " / " FMT_LSIZE " samples\r", epoch + 1, samples_processed_epoch, train_samples_size);
                // fflush(stdout);
            }
            #endif
        }
        println("\nEpoch " FMT_u32 " finished. Processed " FMT_LSIZE " samples.", epoch + 1, samples_processed_epoch);

        // printf("Loss [Sum: %d; Mean: %d (Scale = %d)]\n", loss_sum, loss_sum / (int32_t) (train_samples_size * network->layers[network->num_layers - 1]->activations.height), 0);

        // evaluate(network, X_test, Y_test, test_samples_size);
        evaluate_full(network, X_test, Y_test, test_samples_size, X_train, Y_train, train_samples_size);
    }
    println("\n--- Training Finished (C) ---");
}

void load_batch(int8_t*** X, const Vector8* Y, Matrix8* x_batch, Vector8* y_batch, lsize_t start_idx, lsize_t batch_size, lsize_t features) {
    // free_m8(x_batch);
    *x_batch = init_m8(batch_size, features);
    for (lsize_t i = 0; i < batch_size; i++) {
        memcpy(x_batch->matrix[i], (*X)[start_idx + i], features * sizeof(int8_t));
        // for (lsize_t j = 0; j < features; j++) {
        //     (*x_batch).matrix[i][j] = (*X)[start_idx + i][j];
        // }
    }
    x_batch->scale = 0;

    // free_v8(y_batch);
    *y_batch = init_v8(batch_size);
    memcpy(y_batch->vector, Y->vector + start_idx, batch_size * sizeof(int8_t));
    // for (lsize_t i = 0; i < batch_size; i++) {
    //     (*y_batch).vector[i] = Y->vector[start_idx + i];
    // }
    y_batch->scale = Y->scale;
}

void cleanup(Network* network, int8_t*** X_train, Vector8* Y_train, lsize_t train_samples_size, int8_t*** X_test, Vector8* Y_test, lsize_t test_samples_size) {
    println("\nCleaning up resources...");
    println("Freeing network...");
    free_network(network);
    println("Freeing training features...");
    for(lsize_t i=0; i<train_samples_size; ++i) {
        free(X_train[0][i]);
    }
    free(X_train[0]);
    println("Freeing testing features...");
    for(lsize_t i=0; i<test_samples_size; ++i) {
        free(X_test[0][i]);
    }
    free(X_test[0]);
    println("Freeing labels...");
    free_v8(Y_train);
    free_v8(Y_test);
}