#ifndef DATASET_INT_H
#define DATASET_INT_H

#include "base.h"
#include "network.h"
#include <stdint.h>

__bank(1) void load_train_dataset(int8_t*** X, Vector8** Y, lsize_t* samples_size);

__bank(1) void load_test_dataset(int8_t*** X, Vector8** Y, lsize_t* samples_size);

__bank(1) void load_batch(int8_t*** X, const Vector8* Y, Matrix8* x_batch, Vector8* y_batch, lsize_t start_idx, lsize_t batch_size, lsize_t features);

__bank(1) Vector8* predict(const Network* network, const Matrix8* x_batch);

// sizes' length is (layers_size + 1)
__bank(1) Network* create_network(lsize_t layers_size, const LayerType* kinds, const lsize_t* sizes, lsize_t batch_size);

Network* create_raw_network(lsize_t layers_size, const LayerType* kinds, const lsize_t* sizes, lsize_t batch_size);

__bank(1) void evaluate(const Network* network, int8_t*** X_test, const Vector8* Y_test, lsize_t num_test_samples);

__bank(1) void evaluate_full(const Network* network, int8_t*** X_test, const Vector8* Y_test, lsize_t num_test_samples, int8_t*** X_train, const Vector8* Y_train, lsize_t num_train_samples);

__bank(2) void train_network(const Network* network, int8_t*** X_train, Vector8* Y_train, lsize_t train_samples_size, int8_t*** X_test, Vector8* Y_test, lsize_t test_samples_size, uint32_t epochs);

__bank(1) void cleanup(Network* network, int8_t*** X_train, Vector8* Y_train, lsize_t train_samples_size, int8_t*** X_test, Vector8* Y_test, lsize_t test_samples_size);

__bank(1) void init_weights(Matrix8* weights);

#endif