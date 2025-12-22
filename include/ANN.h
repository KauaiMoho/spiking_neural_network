#ifndef ANN_H
#define ANN_H
#include "Matrix.h"
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>

class ANN {

public:

    enum Activation {
        RELU,
        SIGMOID,
        SOFTMAX
    };

    ANN(std::vector<int> layer_sizes_n, std::vector<Activation> activations_n);
    Matrix forward(const Matrix& input);
    void backprop(const Matrix& init_d_loss);
    void update_weights_biases();
    void clear_grads_and_cache();

    float get_learning_rate() const;
    void set_learning_rate(float l);

    void print_weights(int size = 100) const;
    void print_biases(int size = 100) const;

    static void apply_stable_softmax(Matrix& m);
    static float cross_entropy(const Matrix& truth, const Matrix& preds);
    static float relu(float x);
    static float sigmoid(float x);
    static float deriv_relu(float x);
    static float deriv_sigmoid(float x);

private:

    std::vector<int> layer_sizes;
    std::vector<Activation> activations;
    std::vector<Matrix> weights;
    std::vector<Matrix> biases;
    std::vector<Matrix> z_cache;
    std::vector<Matrix> a_cache;
    std::vector<Matrix> grad_weights;
    std::vector<Matrix> grad_biases;
    float learning_rate = 0.001;

};

#endif