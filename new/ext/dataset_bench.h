#ifndef DATASET_H
#define DATASET_H

#include "../api/network.h"
#include "weights.h"
#include <math.h>
#include <stdint.h>

void load_train_dataset(float*** X_float, Vector8_ext** Y, int* samples_size);

void load_test_dataset(float*** X_float, Vector8_ext** Y, int* samples_size);

void load_batch(float*** X_float, Vector8_ext* Y, Matrix8* x_batch, Vector8* y_batch, int start_idx, int batch_size, int features);

Vector8* predict(Network* network, Matrix8* x_batch);

// sizes' length is (layers_size + 1)
Network* create_network(uint8_t layers_size, LayerType* kinds, uint8_t* sizes, uint8_t batch_size);

void evaluate(Network* network, float*** X_test, Vector8_ext* Y_test, int num_test_samples);

void train_network(Network* network, float*** X_train, Vector8_ext* Y_train, int train_samples_size, float*** X_test, Vector8_ext* Y_test, int test_samples_size, uint32_t epochs);

#endif