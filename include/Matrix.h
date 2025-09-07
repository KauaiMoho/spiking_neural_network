#ifndef MATRIX_H
#define MATRIX_H
using namespace std;

class Matrix {

private:

    //flattened approach
    float* data;
    int* dims;
    int* dists;
    int convert_idx(int* pos);
    int dim_len;
    float data_len;
    
    
public:

    Matrix(int* dims_n, int dim_len, float* data_n);
    Matrix(int* dims_n, int dim_len, int val);
    Matrix matmul(int* this_axes, Matrix other, int*other_axes, int len_a);
    void scmul(float s);
    void add(Matrix a);
    void subtract(Matrix a);
    void transpose(int* axes);
    float get(int* pos);
    float get_index(int i);
    void set(int* pos, float val);
    void set_index(int i, float val);
    int* get_full_dims();
    int get_dims_index(int i);
};

#endif