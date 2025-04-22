#include "dataset_bench.h"
#include <stdlib.h>

Network* create_network(uint8_t layers_size, LayerType* kinds, uint8_t* sizes, uint8_t batch_size) {
    Network* network = init_network(layers_size, batch_size);
    for (uint8_t i = 0; i < layers_size; i++) {
        network->layers[i] = init_layer(batch_size, sizes[i], sizes[i + 1], kinds[i]);
        if (kinds[i] == LINEAR) {
            init_weights_xavier_uniform(&network->layers[i]->weights);
        }
    }
    return network;
}

// --- Helper Function: Predict Class (Handles Exponents) ---
// --- Now handles potentially smaller last batch ---
Vector8* predict(Network* network, Matrix8* x_batch) {
    // network_forward now returns Matrix8 with data and scale
    // Ensure network_forward can handle x_batch->width != network->batch_size if necessary,
    // or temporarily adjust network's expected batch size. Assuming it handles it for now.
    Matrix8 output_activations = network_forward(network, x_batch);

    Vector8* result = (Vector8*) malloc(sizeof(Vector8));
    if (!result) { fprintf(stderr, "Memory allocation failed for prediction result vector\n"); exit(1); }
    result->length = x_batch->width; // Use actual batch size
    result->vector = (int8_t*) malloc(result->length * sizeof(int8_t));
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

// // --- Helper Function: Evaluate Network (Handles Batches, including partial last batch) ---
void evaluate(Network* network, float*** X_test, Vector8_ext* Y_test, int num_test_samples) {
    int batch_size = network->layers[0]->activations.width;
    int input_size = network->layers[0]->weights.height;
    printf("\n--- Evaluating Network (C) ---\n");
    if (num_test_samples == 0) {
        printf("No test samples loaded. Skipping evaluation.\n");
        return;
    }

    int correct_predictions = 0;
    int samples_processed = 0;

    Matrix8 x_batch;
    Vector8 y_batch;

    for (int sample_idx = 0; sample_idx < num_test_samples; sample_idx += batch_size) {
        // <<< CHANGED: Calculate actual batch size, don't skip partial batch
        int current_batch_size = (sample_idx + batch_size <= num_test_samples) ? batch_size : (num_test_samples - sample_idx);
        if (current_batch_size <= 0) continue; // Should not happen with proper loop condition

        // Create quantized batch for evaluation
        // Matrix8 x_batch = create_quantized_batch(X_float_test, sample_idx, current_batch_size, INPUT_SIZE);
        // x_batch.width = current_batch_size;
        // x_batch.height = input_size;
        // x_batch.matrix = &X_test[0][sample_idx];

        // Get corresponding labels for the current batch
        // <<< CHANGED: Ensure v_cpy_range handles length correctly
        // Vector8 y_batch = v_cpy_range(Y_test, sample_idx, sample_idx + current_batch_size); // end index is exclusive
        // y_batch.length = current_batch_size;
        // y_batch.vector = &Y_test[0][sample_idx];

        load_batch(X_test, Y_test, &x_batch, &y_batch, sample_idx, current_batch_size, input_size);

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
        // free_m8(&x_batch);
        // free_v8(&y_batch); // Free the copied labels
    }

    double accuracy = (samples_processed > 0) ? (double)correct_predictions / samples_processed * 100.0 : 0.0;
    printf("Evaluation Complete.\n");
    // <<< CHANGED: Use samples_processed which correctly accounts for partial batches
    printf("Accuracy on test set: %.2f%% (%d / %d correct)\n", accuracy, correct_predictions, samples_processed);
}

void train_network(Network* network, float*** X_train, Vector8_ext* Y_train, int train_samples_size, float*** X_test, Vector8_ext* Y_test, int test_samples_size, uint32_t epochs) {
    int batch_size = network->layers[0]->activations.width;
    int input_size = network->layers[0]->weights.height;
    printf("\n--- Starting Training (C) for %d Epochs ---\n", epochs);
    Matrix8 x_batch;
    Vector8 y_batch;
    for (uint16_t epoch = 0; epoch < epochs; ++epoch) {
        printf("Epoch %d/%d\n", epoch + 1, epochs);
        int samples_processed_epoch = 0;

        for (int sample_idx = 0; sample_idx < train_samples_size; sample_idx += batch_size) {
            int current_batch_size = (sample_idx + batch_size <= train_samples_size) ? batch_size : (train_samples_size - sample_idx);
            if (current_batch_size <= 0) continue;

            load_batch(X_train, Y_train, &x_batch, &y_batch, sample_idx, current_batch_size, input_size);

            // Create quantized batch for training
            // Pass current_batch_size and INPUT_SIZE
            // Matrix8 x_batch = create_quantized_batch(X_train_float, sample_idx, current_batch_size, INPUT_SIZE);

            // x_batch.width = current_batch_size;
            // x_batch.height = input_size;
            // x_batch.matrix = &X_train[0][sample_idx];
            
            // Get corresponding labels for the current batch
            // Ensure v_cpy_range end index is exclusive: [start, end)
            // Vector8 y_batch = v_cpy_range(&Y_train_full, sample_idx, sample_idx + current_batch_size);

            // y_batch.length = current_batch_size;
            // y_batch.vector = &Y_train[0][sample_idx];

            // --- Forward Pass ---
            // network_forward needs to handle the actual batch size in x_batch.width
            Matrix8 out_activations = network_forward(network, &x_batch);
            // We don't need the result directly here, backward pass uses internal state

            // --- Backward Pass & Weight Update ---
            // network_backward needs to handle the actual batch size in y_batch.length
            Matrix8 err_back_to_input = network_backward(network, &y_batch);

            // Free matrices created for the batch
            // free_m8(&x_batch);
            // free_v8(&y_batch);
            free_m8(&out_activations); // Free the result of forward pass
            free_m8(&err_back_to_input); // Free the result of backward pass

            samples_processed_epoch += current_batch_size;
            if ((samples_processed_epoch % 512 == 0) || (sample_idx + current_batch_size >= train_samples_size)) {
                 printf("  Epoch %d: Processed %d / %d samples\r", epoch + 1, samples_processed_epoch, train_samples_size);
                 fflush(stdout);
            }
        }
        printf("\nEpoch %d finished. Processed %d samples.\n", epoch + 1, samples_processed_epoch);

        // Evaluate on test set after each epoch
        evaluate(network, X_test, Y_test, test_samples_size);

    }

    printf("\n--- Training Finished (C) ---\n");
}

Matrix8 to_m8(Matrix8_ext* matrix) {
    Matrix8 result = init_m8(matrix->width, matrix->height);
    for (int i = 0; i < matrix->width; i++) {
        for (int j = 0; j < matrix->height; j++) {
            result.matrix[i][j] = matrix->matrix[i][j];
        }
    }
    return result;
}

Vector8 to_v8(Vector8_ext* vector) {
    Vector8 result = init_v8(vector->length);
    for (int i = 0; i < vector->length; i++) {
        result.vector[i] = vector->vector[i];
    }
    return result;
}

Vector8 v_ext_cpy_range(Vector8_ext* orig, int start, int end) {
    Vector8 result = init_v8(end - start);
    for (int i = start; i < end; i++) {
        result.vector[i - start] = orig->vector[i];
    }
    return result;
}

void load_batch(float*** X_float, Vector8_ext* Y, Matrix8* x_batch, Vector8* y_batch, int start_idx, int batch_size, int features) {
    float** batch_float = malloc(batch_size * sizeof(float*));
    if (!batch_float) { fprintf(stderr, "Memory allocation failed for batch_float pointers\n"); exit(1); }
    for(int i=0; i<batch_size; ++i) {
        batch_float[i] = malloc(features * sizeof(float));
        for (int j = 0; j < features; j++) {
            batch_float[i][j] = X_float[0][start_idx + i][j];
        }
        // Point to the rows in the full float data (avoid deep copy)
    }

    Matrix8_ext m = quantize_float_matrix_adaptive(batch_float, batch_size, features);
    // *x_batch = to_m8(&m);
    x_batch->width = m.width;
    x_batch->height = m.height;
    x_batch->scale = m.scale;
    x_batch->matrix = malloc(m.width * sizeof(int8_t*));
    for (int i = 0; i < m.width; i++) {
        x_batch->matrix[i] = malloc(m.height * sizeof(int8_t));
        for (int j = 0; j < m.height; j++) {
            x_batch->matrix[i][j] = m.matrix[i][j];
        }
    }

    // Vector8 v = v_ext_cpy_range(Y, start_idx, start_idx + batch_size + 1);
    // *y_batch = v;
    y_batch->length = batch_size;
    y_batch->vector = malloc(batch_size * sizeof(int8_t));
    for (int i = 0; i < batch_size; i++) {
        y_batch->vector[i] = Y->vector[start_idx + i];
    }

    for(int i=0; i<batch_size; ++i) {
        free(batch_float[i]);
    }
    free(batch_float);
}