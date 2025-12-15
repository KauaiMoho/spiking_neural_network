#include "ANN.h"
using std::initializer_list;
#include <stdexcept>
#include <algorithm>
#include <cmath>

float ANN::relu(float x) {
    return std::max(0.0f, x);
}

float ANN::sigmoid(float x) {
    return (1 / (1 + std::exp(-x)));
}