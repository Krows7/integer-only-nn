#include "dataset_bench.h"
#include "float_ops.h"
#include "linear.h"
#include "weights.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

Network* create_network(lsize_t layers_size, LayerType* kinds, lsize_t* sizes, lsize_t batch_size) {
    printf("Initializing network...\n");
    Network* network = init_network(layers_size, batch_size);
    printf("Initializing layers...\n");
    for (lsize_t i = 0; i < layers_size; i++) {
        network->layers[i] = init_layer(batch_size, sizes[i], sizes[i + 1], kinds[i]);
        if (kinds[i] == LINEAR) {
            init_weights_xavier_uniform(&network->layers[i]->weights);
        }
    }
    print_network(network);
    return network;
}

Vector8* predict(Network* network, Matrix8* x_batch) {
    Matrix8 output_activations = network_forward(network, x_batch);

    Vector8* result = (Vector8*) malloc(sizeof(Vector8));
    if (!result) { fprintf(stderr, "Memory allocation failed for prediction result vector\n"); exit(1); }
    result->length = x_batch->width;
    result->vector = (int8_t*) malloc(result->length * sizeof(int8_t));
    if (!result->vector) { fprintf(stderr, "Memory allocation failed for prediction result data\n"); exit(1); }
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

void evaluate(Network* network, float*** X_test, Vector8* Y_test, lsize_t num_test_samples) {
    lsize_t batch_size = network->batch_size;
    // lsize_t batch_size = network->layers[0]->activations.width;
    lsize_t input_size = network->layers[0]->weights.height;
    printf("\n--- Evaluating Network (C) ---\n");
    if (num_test_samples == 0) {
        printf("No test samples loaded. Skipping evaluation.\n");
        return;
    }

    int correct_predictions = 0;
    int samples_processed = 0;

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

    double accuracy = (samples_processed > 0) ? (double)correct_predictions / samples_processed * 100.0 : 0.0;

    printf("Evaluation Complete.\n");
    printf("Accuracy on test set: %.2f%% (%d / %d correct)\n", accuracy, correct_predictions, samples_processed);
}

void print_loss(Matrix8* loss) {
    int32_t sum = 0;
    for (lsize_t i = 0; i < loss->width; ++i) {
        for (lsize_t j = 0; j < loss->height; ++j) {
            sum += loss->matrix[j][i];
        }
    }
    printf("Loss [Sum: %d; Mean: %d (Scale = %d)]\n", sum, sum / (loss->width * loss->height), loss->scale);
}

int32_t m8_sum(Matrix8* m) {
    int32_t sum = 0;
    for (lsize_t i = 0; i < m->width; ++i) {
        for (lsize_t j = 0; j < m->height; ++j) {
            if (m->scale >= 0) sum += m->matrix[i][j] << m->scale;
            else sum += m->matrix[i][j] >> -m->scale;
        }
    }
    return sum;
}

void train_network(Network* network, float*** X_train, Vector8* Y_train, lsize_t train_samples_size, float*** X_test, Vector8* Y_test, lsize_t test_samples_size, uint32_t epochs) {
    lsize_t batch_size = network->batch_size;
    // lsize_t batch_size = network->layers[0]->activations.width;
    lsize_t input_size = network->layers[0]->weights.height;
    printf("\n--- Starting Training (C) for %d Epochs ---\n", epochs);
    Matrix8 x_batch;
    Vector8 y_batch;
    for (uint32_t epoch = 0; epoch < epochs; ++epoch) {
        printf("\nEpoch %d/%d\n", epoch + 1, epochs);
        int samples_processed_epoch = 0;
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

            samples_processed_epoch += current_batch_size;
            if ((samples_processed_epoch % 512 == 0) || (sample_idx + current_batch_size >= train_samples_size)) {
                 printf("  Epoch %d: Processed %d / %d samples\r", epoch + 1, samples_processed_epoch, train_samples_size);
                 fflush(stdout);
            }
        }
        printf("\nEpoch %d finished. Processed %d samples.\n", epoch + 1, samples_processed_epoch);

        // printf("Loss [Sum: %d; Mean: %d (Scale = %d)]\n", loss_sum, loss_sum / (int32_t) (train_samples_size * network->layers[network->num_layers - 1]->activations.height), 0);

        evaluate(network, X_test, Y_test, test_samples_size);
    }

    printf("\n--- Training Finished (C) ---\n");
}

Matrix8 to_m8(Matrix8* matrix) {
    Matrix8 result = init_m8(matrix->width, matrix->height);
    for (lsize_t i = 0; i < matrix->width; i++) {
        for (lsize_t j = 0; j < matrix->height; j++) {
            result.matrix[i][j] = matrix->matrix[i][j];
        }
    }
    return result;
}

Vector8 to_v8(Vector8* vector) {
    Vector8 result = init_v8(vector->length);
    for (lsize_t i = 0; i < vector->length; i++) {
        result.vector[i] = vector->vector[i];
    }
    return result;
}

Vector8 v_ext_cpy_range(Vector8* orig, lsize_t start, lsize_t end) {
    Vector8 result = init_v8(end - start);
    for (lsize_t i = start; i < end; i++) {
        result.vector[i - start] = orig->vector[i];
    }
    return result;
}

void load_batch(float*** X_float, Vector8* Y, Matrix8* x_batch, Vector8* y_batch, lsize_t start_idx, lsize_t batch_size, lsize_t features) {
    float** batch_float = malloc(batch_size * sizeof(float*));
    if (!batch_float) { fprintf(stderr, "Memory allocation failed for batch_float pointers\n"); exit(1); }
    for(lsize_t i=0; i<batch_size; ++i) {
        batch_float[i] = malloc(features * sizeof(float));
        for (lsize_t j = 0; j < features; j++) {
            batch_float[i][j] = X_float[0][start_idx + i][j];
        }
    }

    Matrix8 m = quantize_float_matrix_adaptive(batch_float, batch_size, features);
    x_batch->width = m.width;
    x_batch->height = m.height;
    x_batch->scale = m.scale;
    x_batch->matrix = malloc(m.width * sizeof(int8_t*));
    for (lsize_t i = 0; i < m.width; i++) {
        x_batch->matrix[i] = malloc(m.height * sizeof(int8_t));
        for (lsize_t j = 0; j < m.height; j++) {
            x_batch->matrix[i][j] = m.matrix[i][j];
        }
    }

    y_batch->length = batch_size;
    y_batch->vector = malloc(batch_size * sizeof(int8_t));
    for (lsize_t i = 0; i < batch_size; i++) {
        y_batch->vector[i] = Y->vector[start_idx + i];
    }

    for(lsize_t i=0; i<batch_size; ++i) {
        free(batch_float[i]);
    }
    free(batch_float);
    free_m8(&m);
}

void cleanup(Network* network, float*** X_train, Vector8* Y_train, lsize_t train_samples_size, float*** X_test, Vector8* Y_test, lsize_t test_samples_size) {
    printf("\nCleaning up resources...\n");
    printf("Freeing network...\n");
    free_network(network);
    printf("Freeing training features...\n");
    for(lsize_t i=0; i<train_samples_size; ++i) {
        free(X_train[0][i]);
    }
    free(X_train[0]);
    printf("Freeing testing features...\n");
    for(lsize_t i=0; i<test_samples_size; ++i) {
        free(X_test[0][i]);
    }
    free(X_test[0]);
    printf("Freeing labels...\n");
    free_v8(Y_train);
    free_v8(Y_test);
}