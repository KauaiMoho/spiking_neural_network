#ifndef ANN_H
#define ANN_H
#include "Matrix.h"
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>

class ANN {

enum Activation {
    RELU,
    SIGMOID
};

private:

    std::vector<int> layer_sizes;
    std::vector<Activation> activations;
    std::vector<Matrix> weights;
    std::vector<Matrix> biases;
    
public:

    ANN(std::vector<int> layer_sizes_n, std::vector<Activation> activations_n);
    Matrix forward(const Matrix& input);
    static float relu(float x);
    static float sigmoid(float x);
    static float deriv_relu(float x);
    static float deriv_sigmoid(float x);
    
};

#endif