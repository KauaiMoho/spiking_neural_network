#include "../include/Matrix.h"
#include <iostream>
using namespace std;

int main() {
    Matrix::set_CUDA(false);
    int dim1[] =       {7, 9, 1, 9, 2};
    int dim2[] = {4, 5, 7, 1, 6, 2, 3};
    Matrix m = Matrix(dim1, 5, 10);
    Matrix b = Matrix(dim2, 7, 12);
    Matrix c = m.matmul(b);
    c.print_dims();
    cout << c.get({0, 0, 2});
    cout << '\n';
}