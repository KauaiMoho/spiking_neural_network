#include "../include/Matrix.h"
#include <iostream>
using namespace std;

int main() {

    int numbers[] = {12, 3, 4, 9};
    int idx[] = {6, 1, 2, 1};
    Matrix m = Matrix(numbers, 4, 10);
    m.set(idx, 12);
    cout << m.get(idx);
}