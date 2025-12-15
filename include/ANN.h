#ifndef ANN_H
#define ANN_H
#include <initializer_list>
#include "Matrix.h"
using std::initializer_list;
using namespace std;

class ANN {

private:

    float* layers;
    Matrix weights;
    Matrix biases;
    
public:
    
    float relu(float x);
    float sigmoid(float x);
    
};

#endif