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

    Matrix(vector<int> dims, vector<float> data);
    Matrix(vector<int> dims);
    Matrix matmul(vector<int> this_axes, Matrix other, vector<int> other_axes);
    void scmul(float s);
    void add(Matrix a);
    void subtract(Matrix a);
    void transpose();
    float get(vector<int> pos);
    void set(vector<int> pos, float val);
    vector<int> get_full_dims();
    int get_dims_index(int i);
    float get_index(int i);
};

#endif