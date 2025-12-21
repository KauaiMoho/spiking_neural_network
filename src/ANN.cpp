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

    //Can change in future if we want to generalize this class.
    for (size_t i = 0; i < layer_sizes.size(); ++i) {
        if (i != layer_sizes.size() - 1 && activations[i] == SOFTMAX) {
            throw std::invalid_argument("Hidden layers cannot have SOFTMAX.");
        }
        if (i == layer_sizes.size() - 1 && activations[i] != SOFTMAX) {
            throw std::invalid_argument("Final layer must have SOFTMAX for cross_entropy backprop.");
        }
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

    z_cache.clear();
    a_cache.clear();
    
    Matrix ret = input.clone();
    for (size_t i = 0; i < layer_sizes.size(); ++i) {
        a_cache.push_back(ret.clone());
        ret = ret.matmul(weights[i]);
        if (i != layer_sizes.size() - 1) {
            z_cache.push_back(ret.clone());
        }
        ret.add_inplace(biases[i]);
        if (activations[i] == RELU) {
            ret.apply_inplace(relu);
        } else if (activations[i] == SIGMOID) {
            ret.apply_inplace(sigmoid);
        } else if (activations[i] == SOFTMAX) {
            apply_softmax(ret);
        } else {
            throw std::invalid_argument("Invalid activation.");
        }
    }

    return ret;
}

std::tuple<std::vector<Matrix>, std::vector<Matrix>> ANN::backprop(const Matrix& init_d_loss) const {
    Matrix d_loss = init_d_loss.clone();

    std::vector<Matrix> d_weights;
    std::vector<Matrix> d_biases;

    for (size_t i = layer_sizes.size() - 1; i > -1; i--) {
        d_weights.push_back(a_cache[i].transpose2d().matmul(d_loss));
        d_biases.push_back(d_loss.sum_rows());
        d_loss = d_loss.matmul(weights[i].transpose2d());
        if (activations[i] == RELU) {
            d_loss.emul_inplace(z_cache[i-1].apply(deriv_relu));
        } else if (activations[i] == SIGMOID) {
            d_loss.emul_inplace(z_cache[i-1].apply(deriv_sigmoid));
        } else {
            throw std::invalid_argument("Invalid activation for backprop.");
        }
    }

    return std::make_tuple(d_weights, d_biases);

}

void ANN::update_weights(std::vector<Matrix> d_weights,  std::vector<Matrix> d_biases) {
    if (d_weights.size() != weights.size() || d_biases.size() != biases.size()) {
        throw std::invalid_argument("Invalid input size!");
    }

    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i].add_inplace(d_weights[i].scmul(-learning_rate));
        biases[i].add_inplace(d_biases[i].scmul(-learning_rate));
    }
}

//Only for a 1D vectors.
float ANN::cross_entropy(const Matrix& truth, const Matrix& preds) const {
    if (truth.get_dim_len() == 1 && preds.get_dim_len() == 1) {
        if (truth.get_dims_index(0) == preds.get_dims_index(0)) {
            float loss = 0;
            for (int i = 0; i < truth.get_dims_index(0); ++i) {
                loss += truth.get({i}) - log(preds.get({i}));
            }
            return -loss;
        } else {
            throw std::invalid_argument("Incompatible shapes for cross_entropy loss!");
        }
    } else {
        throw std::invalid_argument("Invalid dimensions for cross_entropy loss!");
    }
}

//Only for a 1D vector.
void ANN::apply_softmax(Matrix& m) const {
    if (m.get_dim_len() == 1) {
        float sum = 0;
        for (int i = 0; i < m.get_dims_index(0); ++i) {
            float val = exp(m.get({i}));
            sum += val;
            m.set({i}, val);
        }
        for (int i = 0; i < m.get_dims_index(0); ++i) {
            m.set({i}, m.get({i})/sum);
        }
    } else {
        throw std::invalid_argument("Invalid dimension for softmax!");
    }
}

float ANN::get_learning_rate() const {
    return learning_rate;
}

void ANN::set_learning_rate(float l) {
    learning_rate = l;
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