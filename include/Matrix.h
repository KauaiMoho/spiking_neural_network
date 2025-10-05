#ifndef MATRIX_H
#define MATRIX_H
#include <initializer_list>
using std::initializer_list;

class Matrix {

private:

    //flattened approach
    float* data;
    int* dims;
    int* dists;
    int dim_len;
    float data_len;
    static bool cuda;
    
    int convert_idx(initializer_list<int> pos);
    void matmul_cuda(float* A, float* B, float* C, int n, int m, int k);
    void matmul_cpu(float* A, float* B, float* C, int n, int m, int k);
    
public:

    
    Matrix(int* dims_n, int dim_len, float* data_n);
    Matrix(int* dims_n, int dim_len, int val);
    Matrix matmul(Matrix other);
    Matrix clone();
    void scmul(float s);
    void add(Matrix a);
    void subtract(Matrix a);
    void transpose(int* axes);
    float get(initializer_list<int> pos);
    float get_index(int i);
    void set(initializer_list<int> pos, float val);
    void set_index(int i, float val);
    void broadcast(int* dim, int dim_len);
    int* get_broadcasted_strides(int* dims_new, int dim_len_new);
    int* get_full_dims();
    int get_dims_index(int i);
    int get_dim_len();
    float* get_data();
    static void set_CUDA(bool c);
    static bool get_CUDA();
    
};

extern "C" void matmul_cuda(const float* A, const float* B, float* C, int n, int m, int k);

#endif