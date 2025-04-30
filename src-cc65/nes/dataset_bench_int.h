#ifndef DATASET_INT_H
#define DATASET_INT_H

#include "network.h"

void load_train_dataset(int8_t*** X, Vector8** Y, lsize_t* samples_size);

void load_test_dataset(int8_t*** X, Vector8** Y, lsize_t* samples_size);

void load_batch(int8_t*** X, Vector8* Y, Matrix8* x_batch, Vector8* y_batch, lsize_t start_idx, lsize_t batch_size, lsize_t features);

Vector8* predict(Network* network, Matrix8* x_batch);

// sizes' length is (layers_size + 1)
Network* create_network(lsize_t layers_size, LayerType* kinds, lsize_t* sizes, lsize_t batch_size);

void evaluate(Network* network, int8_t*** X_test, Vector8* Y_test, lsize_t num_test_samples);

void train_network(Network* network, int8_t*** X_train, Vector8* Y_train, lsize_t train_samples_size, int8_t*** X_test, Vector8* Y_test, lsize_t test_samples_size, uint32_t epochs);

void cleanup(Network* network, int8_t*** X_train, Vector8* Y_train, lsize_t train_samples_size, int8_t*** X_test, Vector8* Y_test, lsize_t test_samples_size);

void init_weights(Matrix8* weights);

#endif