#include "../include/Matrix.h" // this contains the Matrix class definition
#include "../include/ANN.h" // this contains the ANN class definition
#include <iostream>
#include <vector>
#include <stdexcept>

int main() {
    std::vector<int> sizes = {784, 128, 64, 10};
    std::vector<ANN::Activation> activations = {
        ANN::Activation::RELU, 
        ANN::Activation::RELU, 
        ANN::Activation::SOFTMAX
    };
    ANN model = ANN(sizes, activations);

    model.print_weights();
    model.print_biases();

    int batch = 1;
    int dim_input[] = {batch, 784};
    int dim_output[] = {batch, 10};

    //Same size as MNIST, but random for while testing code run only (not functionality)
    //Later change to be dataloaded
    Matrix test_input = Matrix(dim_input, 2);

    Matrix test_correct_label = Matrix(dim_output, 2, (float)0);
    //Pretend like the correct label is the letter C
    test_correct_label.set({0, 2}, 1);

    Matrix test_output = model.forward(test_input);

    std::cout << "Loss: " << ANN::cross_entropy(test_correct_label, test_output) << "\n";

    //Crossentropy Loss + Softmax Derivitive - A given for this implementation to avoid Jacobian math.
    Matrix test_d_loss = test_output.add(test_correct_label.scmul(-1));
    model.backprop(test_d_loss);
    model.update_weights_biases();

    model.print_weights();
    model.print_biases();

    std::cout << "FINISHED!\n";

}