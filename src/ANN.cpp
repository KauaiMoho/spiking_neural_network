#include "ANN.h"

ANN::ANN(std::vector<int> layer_sizes_n, std::vector<Activation> activations_n) : 
    layer_sizes(std::move(layer_sizes_n)), activations(std::move(activations_n)) 
{
    if (layer_sizes.size() != activations.size()) {
        throw std::invalid_argument("Layer counts must match activation counts!");
    }

    if (layer_sizes.size() == 0) {
        throw std::invalid_argument("Cannot have empty ANN!");
    }

    int prev_size = layer_sizes[0];
    for (size_t i = 1; i < layer_sizes.size(); ++i) {
        int curr_size = layer_sizes[i];
        int dims_weights[] = {prev_size, curr_size};
        int dims_biases[] = {curr_size};
        weights.push_back(Matrix(dims_weights, 2));
        biases.push_back(Matrix(dims_biases, 1, (float)0));
        prev_size = curr_size;
    }
}

Matrix ANN::forward(const Matrix& input) {
    if (input.get_dim_len() != 2 || input.get_dims_index(1) != layer_sizes[0]) {
        throw std::invalid_argument("Invalid input size!");
    }
    
    Matrix ret = input.clone();
    for (size_t i = 0; i < layer_sizes.size(); ++i) {
        ret = ret.matmul(weights[i]);
        ret.add_inplace(biases[i]);
        if (activations[i] == RELU) {
            ret.apply_inplace(relu);
        } else {
            ret.apply_inplace(sigmoid);
        }
    }

    return ret;
}

float ANN::relu(float x) {
    return std::max(0.0f, x);
}

float ANN::sigmoid(float x) {
    return (1 / (1 + std::exp(-x)));
}

float ANN::deriv_relu(float x) {
    if (x <= 0) {
        return 0;
    }
    return 1;
}

float ANN::deriv_sigmoid(float x) {
    float sig = sigmoid(x);
    return sig * (1 - sig);
}