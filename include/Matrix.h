#ifndef MATRIX_H
#define MATRIX_H
#include <vector>
using namespace std;

class Matrix {

private:

    //flattened approach
    vector<float> data;
    vector<int> dims;
    vector<int> dists;
    int get_idx(vector<int> pos);
    
    
public:

    Matrix(vector<int> dims);
    Matrix matmul(Matrix a);
    Matrix scmul(Matrix a);
    Matrix add(Matrix a);
    Matrix subtract(Matrix a);
    Matrix transpose();
    float get(vector<int> pos);
    void set(vector<int> pos, float val);
};

#endif