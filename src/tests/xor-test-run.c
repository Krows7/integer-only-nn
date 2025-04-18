#include "../api/network.h"
#include <linux/limits.h>

int main(void) {
    Matrix8 X = init_m8(4, 2);
    X.matrix[0][0] = 0;
    X.matrix[0][1] = 0;
    X.matrix[1][0] = 0;
    X.matrix[1][1] = 1;
    X.matrix[2][0] = 1;
    X.matrix[2][1] = 0;
    X.matrix[3][0] = 1;
    X.matrix[3][1] = 1;
    
    // Matrix8 y = init_m8(4, 1);
    // y.matrix[0][0] = 0;
    // y.matrix[1][0] = 1;
    // y.matrix[2][0] = 1;
    // y.matrix[3][0] = 0;

    Vector8 Y = init_v8(4);
    Y.vector[0] = 0;
    Y.vector[1] = 1;
    Y.vector[2] = 1;
    Y.vector[3] = 0;
    
    Network* network = init_network(2, 1);
    network->layers[0] = init_layer(1, 2, 2, LINEAR, network->layers[1]);
    network->layers[1] = init_layer(1, 2, 2, LINEAR, NULL);
    
    Matrix8 x = init_m8(1, 2);
    Vector8 y = init_v8(1);

    for (uint8_t i = 0; i < 5; ++i) {
        for (uint8_t j = 0; j < 4; ++j) {
            x.matrix[0][0] = X.matrix[j][0];
            x.matrix[0][1] = X.matrix[j][1];
            y.vector[0] = Y.vector[j];

            // Matrix8 out = network_forward(network, &x);
            network_forward(network, &x);

            // Matrix8 loss = loss_gradient(&out, &y);

            // print_matrix8(&loss, "Loss");

            network_backward(network, &y);
        }
    }

    print_layer(network->layers[1], "Output");
}