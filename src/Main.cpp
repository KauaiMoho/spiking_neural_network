#include "../include/Matrix.h"
#include <iostream>
using namespace std;

int main() {
    Matrix::set_CUDA(false);
    // int dim1_[] =       {7, 9, 1, 9, 2};
    // int dim2_[] = {4, 5, 7, 1, 6, 2, 3};
    // Matrix m = Matrix(dim1_, 5, 10);
    // Matrix b = Matrix(dim2_, 7, 12);
    // Matrix c = m.matmul(b);

    int dim1[] = {2, 1, 2};
    int dim2[] = {2, 2, 1};

    float* dataM = new float[4];
    float* dataN = new float[4];

    for (int i = 0; i < 4; i++) dataM[i] = static_cast<float>(i + 1);
    for (int i = 0; i < 4; i++) dataN[i] = static_cast<float>(i + 5);

    Matrix M = Matrix(dim1, 3, dataM);
    Matrix N = Matrix(dim2, 3, dataN);
    
    Matrix P = M.matmul(N);
    P.print_dims(); 

    return 0;
}