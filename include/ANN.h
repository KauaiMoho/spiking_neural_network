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
    void apply_softmax(Matrix& m) const;
    float cross_entropy(const Matrix& truth, const Matrix& preds) const;
    std::tuple<std::vector<Matrix>, std::vector<Matrix>> backprop(const Matrix& init_d_loss) const;
    void update_weights(std::vector<Matrix> d_weights,  std::vector<Matrix> d_biases);

    float get_learning_rate() const;
    void set_learning_rate(float l);

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
    float learning_rate = 0.001;

};

#endif