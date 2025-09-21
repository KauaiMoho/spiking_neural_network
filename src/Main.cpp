#include "../include/Matrix.h"
#include <iostream>
using namespace std;

int main() {
    Matrix::set_CUDA(false);
    int dim1[] = {9, 11};
    int dim2[] = {11, 13};
    Matrix m = Matrix(dim1, 2, 10);
    Matrix b = Matrix(dim2, 2, 12);
    Matrix c = m.matmul(b);
    cout << c.get_dims_index(0);
    cout << '\n';
    cout << c.get_dims_index(1);
    cout << '\n';
    cout << c.get({0, 2});
}