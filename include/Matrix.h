#ifndef MATRIX_H
#define MATRIX_H
using namespace std;

class Matrix {

private:

    //flattened approach
    float* data;
    int* dims;
    int* dists;
    int get_idx(int* pos);
    int dim_len;
    float data_len;
    
    
public:

    Matrix(int* dims_n, int dim_len, float* data_n, int data_len);
    Matrix(int* dims_n, int dim_len, int val);
    Matrix matmul(int* this_axes, int len_this_a, Matrix other, int*other_axes, int len_other_a);
    void scmul(float s);
    void add(Matrix a);
    void subtract(Matrix a);
    void transpose();
    float get(int* pos);
    void set(int* pos, float val);
    int* get_full_dims();
    int get_dims_index(int i);
    float get_index(int i);
};

#endif