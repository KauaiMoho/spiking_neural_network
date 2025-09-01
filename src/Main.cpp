#include "../include/Matrix.h"
#include <iostream>
using namespace std;

int main() {
    Matrix m = Matrix({12, 2, 4, 9});
    m.set({6, 1, 2, 7}, 10);
    cout << m.get({6, 1, 2, 7});
}