#include "dataset_bench_int.h"
#include "base.h"
#include "linear.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// integer square-root (any standard 32-bit routine will do)
uint32_t isqrt32(uint32_t x) {
    uint32_t res = 0, bit = 1u << 30;
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
    uint32_t ratio = (6 << 16) / sum;
    uint32_t limit = isqrt32(ratio);
    if (limit > 255) limit = 255;
    weights->scale = -8;

    for (lsize_t i = 0; i < weights->width; i++) {
        for (lsize_t j = 0; j < weights->height; j++) {
            int16_t r16 = rand16();
            int16_t v = (r16 * (int)limit) >> 15;
            weights->matrix[i][j] = (int8_t)v;
        }
    }
}

void init_weights(Matrix8 *weights) {
    init_xavier(weights);
}

Network* create_network(lsize_t layers_size, LayerType* kinds, lsize_t* sizes, lsize_t batch_size) {
    println("Initializing network...");
    Network* network = init_network(layers_size, batch_size);
    println("Initializing layers...");
    for (lsize_t i = 0; i < layers_size; i++) {
        network->layers[i] = init_layer(batch_size, sizes[i], sizes[i + 1], kinds[i]);
        if (kinds[i] == LINEAR) {
            init_weights(&network->layers[i]->weights);
        }
    }
    #if PRINT_ALLOWED > 0
    print_network(network);
    #endif
    return network;
}

Vector8* predict(Network* network, Matrix8* x_batch) {
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
        for (lsize_t i = 0; i < output_activations.height; ++i) {
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

void evaluate(Network* network, int8_t*** X_test, Vector8* Y_test, lsize_t num_test_samples) {
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

    #if PRINT_ALLOWED > 0
    uint8_t accuracy = (uint16_t) correct_predictions * 100 / samples_processed;
    #endif

    println("Evaluation Complete.");
    println("Accuracy on test set: ~%d%% (%d / %d correct)", accuracy, correct_predictions, samples_processed);
}

void print_loss(Matrix8* loss) {
    int32_t sum = 0;
    for (lsize_t i = 0; i < loss->width; ++i) {
        for (lsize_t j = 0; j < loss->height; ++j) {
            sum += loss->matrix[j][i];
        }
    }
    println("Loss [Sum: %d; Mean: %d (Scale = %d)]", sum, sum / (loss->width * loss->height), loss->scale);
}

void train_network(Network* network, int8_t*** X_train, Vector8* Y_train, lsize_t train_samples_size, int8_t*** X_test, Vector8* Y_test, lsize_t test_samples_size, uint32_t epochs) {
    lsize_t batch_size = network->batch_size;
    lsize_t input_size = network->layers[0]->weights.height;
    println("\n--- Starting Training (C) for %d Epochs ---", epochs);
    Matrix8 x_batch;
    Vector8 y_batch;
    for (uint32_t epoch = 0; epoch < epochs; ++epoch) {
        println("\nEpoch %d/%d", epoch + 1, epochs);
        #if PRINT_ALLOWED > 0
        int samples_processed_epoch = 0;
        #endif
        // int32_t loss_sum = 0;

        for (lsize_t sample_idx = 0; sample_idx < train_samples_size; sample_idx += batch_size) {
            lsize_t current_batch_size = (sample_idx + batch_size <= train_samples_size) ? batch_size : (train_samples_size - sample_idx);
            if (current_batch_size <= 0) continue;

            load_batch(X_train, Y_train, &x_batch, &y_batch, sample_idx, current_batch_size, input_size);

            Matrix8 out_activations = network_forward(network, &x_batch);

            // Matrix8 loss = loss_gradient(&out_activations, &y_batch);

            // loss_sum += m8_sum(&loss);

            // print_loss(&loss);

            Matrix8 err_back_to_input = network_backward(network, &y_batch);

            free_m8(&x_batch);
            free_v8(&y_batch);
            free_m8(&out_activations);
            free_m8(&err_back_to_input);

            #if PRINT_ALLOWED > 0
            samples_processed_epoch += current_batch_size;
            if ((samples_processed_epoch % 512 == 0) || (sample_idx + current_batch_size >= train_samples_size)) {
                 printf("  Epoch %d: Processed %d / %d samples\r", epoch + 1, samples_processed_epoch, train_samples_size);
                 fflush(stdout);
            }
            #endif
        }
        println("\nEpoch %d finished. Processed %d samples.", epoch + 1, samples_processed_epoch);

        // printf("Loss [Sum: %d; Mean: %d (Scale = %d)]\n", loss_sum, loss_sum / (int32_t) (train_samples_size * network->layers[network->num_layers - 1]->activations.height), 0);

        evaluate(network, X_test, Y_test, test_samples_size);
    }
    println("\n--- Training Finished (C) ---");
}

void load_batch(int8_t*** X, Vector8* Y, Matrix8* x_batch, Vector8* y_batch, lsize_t start_idx, lsize_t batch_size, lsize_t features) {
    // free_m8(x_batch);
    *x_batch = init_m8(batch_size, features);
    for (lsize_t i = 0; i < batch_size; i++) {
        for (lsize_t j = 0; j < features; j++) {
            (*x_batch).matrix[i][j] = X[0][start_idx + i][j];
        }
    }

    // free_v8(y_batch);
    *y_batch = init_v8(batch_size);
    for (lsize_t i = 0; i < batch_size; i++) {
        (*y_batch).vector[i] = Y->vector[start_idx + i];
    }
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