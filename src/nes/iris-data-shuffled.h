// Generated by script. Do not edit manually.
#ifndef IRIS_DATA_H
#define IRIS_DATA_H

#include <stdint.h> // For int8_t

#define IRIS_SAMPLES 150
#define IRIS_FEATURES 4

// Iris Features (X): int8_t[IRIS_SAMPLES][IRIS_FEATURES]
// Original float values multiplied by 10.0 and rounded/clamped to int8_t.
const int8_t iris_X[IRIS_SAMPLES][IRIS_FEATURES] = {
    {58, 40, 12, 2},
    {50, 30, 16, 2},
    {48, 31, 16, 2},
    {76, 30, 66, 21},
    {61, 30, 46, 14},
    {57, 25, 50, 20},
    {55, 35, 13, 2},
    {50, 20, 35, 10},
    {56, 30, 41, 13},
    {62, 28, 48, 18},
    {48, 30, 14, 3},
    {57, 30, 42, 12},
    {58, 28, 51, 24},
    {51, 38, 15, 3},
    {51, 37, 15, 4},
    {46, 31, 15, 2},
    {49, 24, 33, 10},
    {68, 30, 55, 21},
    {51, 38, 19, 4},
    {50, 34, 16, 4},
    {55, 26, 44, 12},
    {43, 30, 11, 1},
    {50, 33, 14, 2},
    {65, 30, 52, 20},
    {64, 31, 55, 18},
    {44, 30, 13, 2},
    {77, 28, 67, 20},
    {68, 32, 59, 23},
    {47, 32, 13, 2},
    {55, 23, 40, 13},
    {59, 30, 42, 15},
    {77, 30, 61, 23},
    {54, 39, 17, 4},
    {64, 29, 43, 13},
    {60, 30, 48, 18},
    {63, 25, 50, 19},
    {64, 32, 45, 15},
    {54, 37, 15, 2},
    {61, 30, 49, 18},
    {60, 22, 40, 10},
    {67, 25, 58, 18},
    {71, 30, 59, 21},
    {49, 31, 15, 1},
    {60, 27, 51, 16},
    {59, 32, 48, 18},
    {67, 31, 44, 14},
    {65, 28, 46, 15},
    {63, 27, 49, 18},
    {65, 30, 55, 18},
    {55, 25, 40, 13},
    {58, 26, 40, 12},
    {49, 25, 45, 17},
    {62, 29, 43, 13},
    {49, 30, 14, 2},
    {55, 24, 37, 10},
    {63, 25, 49, 15},
    {69, 32, 57, 23},
    {68, 28, 48, 14},
    {67, 30, 50, 17},
    {77, 38, 67, 22},
    {61, 28, 40, 13},
    {52, 34, 14, 2},
    {72, 32, 60, 18},
    {58, 27, 39, 12},
    {57, 29, 42, 13},
    {63, 29, 56, 18},
    {50, 32, 12, 2},
    {51, 35, 14, 3},
    {72, 36, 61, 25},
    {50, 35, 13, 3},
    {51, 25, 30, 11},
    {63, 28, 51, 15},
    {61, 28, 47, 12},
    {52, 35, 15, 2},
    {56, 27, 42, 13},
    {49, 31, 15, 1},
    {54, 30, 45, 15},
    {69, 31, 51, 23},
    {57, 28, 45, 13},
    {57, 26, 35, 10},
    {58, 27, 51, 19},
    {51, 34, 15, 2},
    {51, 33, 17, 5},
    {50, 23, 33, 10},
    {65, 30, 58, 22},
    {65, 32, 51, 20},
    {49, 31, 15, 1},
    {69, 31, 49, 15},
    {54, 39, 13, 4},
    {67, 30, 52, 23},
    {72, 30, 58, 16},
    {60, 34, 45, 16},
    {63, 34, 56, 24},
    {60, 29, 45, 15},
    {59, 30, 51, 18},
    {48, 30, 14, 1},
    {73, 29, 63, 18},
    {64, 27, 53, 19},
    {57, 28, 41, 13},
    {46, 36, 10, 2},
    {66, 30, 44, 14},
    {56, 29, 36, 13},
    {61, 29, 47, 14},
    {50, 36, 14, 2},
    {55, 24, 38, 11},
    {58, 27, 41, 10},
    {74, 28, 61, 19},
    {57, 38, 17, 3},
    {45, 23, 13, 3},
    {62, 22, 45, 15},
    {62, 34, 54, 23},
    {67, 31, 47, 15},
    {50, 35, 16, 6},
    {64, 32, 53, 23},
    {54, 34, 15, 4},
    {56, 28, 49, 20},
    {79, 38, 64, 20},
    {56, 30, 45, 15},
    {63, 23, 44, 13},
    {61, 26, 56, 14},
    {58, 27, 51, 19},
    {63, 33, 60, 25},
    {64, 28, 56, 22},
    {51, 35, 14, 2},
    {70, 32, 47, 14},
    {63, 33, 47, 16},
    {60, 22, 50, 15},
    {55, 42, 14, 2},
    {46, 34, 14, 3},
    {77, 26, 69, 23},
    {67, 33, 57, 25},
    {66, 29, 46, 13},
    {44, 29, 14, 2},
    {54, 34, 17, 2},
    {56, 25, 39, 11},
    {48, 34, 16, 2},
    {52, 27, 39, 14},
    {48, 34, 19, 2},
    {64, 28, 56, 21},
    {67, 31, 56, 24},
    {44, 32, 13, 2},
    {46, 32, 14, 2},
    {51, 38, 16, 2},
    {47, 32, 16, 2},
    {52, 41, 15, 1},
    {69, 31, 54, 21},
    {57, 44, 15, 4},
    {67, 33, 57, 21},
    {50, 34, 15, 2},
    {53, 37, 15, 2}
};

// Iris Class Labels (Y): int8_t[IRIS_SAMPLES]
// Iris-setosa=0, Iris-versicolor=1, Iris-virginica=2
const int8_t iris_Y[IRIS_SAMPLES] = {
    0, 0, 0, 2, 1, 2, 0, 1, 1, 2, 0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 
    1, 0, 0, 2, 2, 0, 2, 2, 0, 1, 1, 2, 0, 1, 2, 2, 1, 0, 2, 1, 
    2, 2, 0, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 0, 1, 1, 2, 1, 1, 2, 
    1, 0, 2, 1, 1, 2, 0, 0, 2, 0, 1, 2, 1, 0, 1, 0, 1, 2, 1, 1, 
    2, 0, 0, 1, 2, 2, 0, 1, 0, 2, 2, 1, 2, 1, 2, 0, 2, 2, 1, 0, 
    1, 1, 1, 0, 1, 1, 2, 0, 0, 1, 2, 1, 0, 2, 0, 2, 2, 1, 1, 2, 
    2, 2, 2, 0, 1, 1, 2, 0, 0, 2, 2, 1, 0, 0, 1, 0, 1, 0, 2, 2, 
    0, 0, 0, 0, 0, 2, 0, 2, 0, 0
};

#endif // IRIS_DATA_H
