#ifndef DATASET_H
#define DATASET_H

#include "network.h"

void load_train_dataset(float*** X_float, Vector8** Y, lsize_t* samples_size);

void load_test_dataset(float*** X_float, Vector8** Y, lsize_t* samples_size);

void load_batch(float*** X_float, Vector8* Y, Matrix8* x_batch, Vector8* y_batch, lsize_t start_idx, lsize_t batch_size, lsize_t features);

Vector8* predict(Network* network, Matrix8* x_batch);

// sizes' length is (layers_size + 1)
Network* create_network(lsize_t layers_size, LayerType* kinds, lsize_t* sizes, lsize_t batch_size);

void evaluate(Network* network, float*** X_test, Vector8* Y_test, lsize_t num_test_samples);

void evaluate_full(Network* network, float*** X_test, Vector8* Y_test, lsize_t num_test_samples, float*** X_train, Vector8* Y_train, lsize_t num_train_samples);

void train_network(Network* network, float*** X_train, Vector8* Y_train, lsize_t train_samples_size, float*** X_test, Vector8* Y_test, lsize_t test_samples_size, uint32_t epochs);

void cleanup(Network* network, float*** X_train, Vector8* Y_train, lsize_t train_samples_size, float*** X_test, Vector8* Y_test, lsize_t test_samples_size);

#endif