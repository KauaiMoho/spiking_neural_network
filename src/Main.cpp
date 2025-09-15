#include "../include/Matrix.h"
#include <iostream>
using namespace std;

int main() {

    int dim1[] = {13};
    int dim2[] = {11, 13};
    Matrix m = Matrix(dim1, 1, 10);
    Matrix b = Matrix(dim2, 2, 12);
    Matrix c = m.matmul(b);
    cout << c.get({2});
}