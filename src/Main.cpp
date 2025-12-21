#include "../include/Matrix.h" // this contains the Matrix class definition
#include "../include/ANN.h" // this contains the ANN class definition
#include <iostream>
#include <vector>
#include <stdexcept>

int main() {
    std::vector<int> sizes = {784, 128, 64, 28};
    std::vector<ANN::Activation> activations = {
        ANN::Activation::SIGMOID,
        ANN::Activation::RELU, 
        ANN::Activation::RELU, 
        ANN::Activation::SOFTMAX
    };
    ANN model = ANN(sizes, activations);

    

}