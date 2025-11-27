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
    int data_len;
    static bool cuda;
    
    int convert_idx(initializer_list<int> pos) const; 
    void matmul_cuda(float* A, float* B, float* C, int n, int m, int k);
    void matmul_cpu_batched(float* A, float* B, float* C, int n, int m, int k, int z);
    void matmul_cpu(float* A, float* B, float* C, int n, int m, int k);
    void simd_transpose(float* A, float* B, int n, int m); //INTERNAL USE ONLY
    int* get_dims_clone();
    int* get_dists_clone();
    void set_dim_len(int dim_len_n); // UNCHECKED
    void set_dims(int* dims_n); // UNCHECKED
    void set_dists(int* dists_n); // UNCHECKED
    
public:
    
    Matrix(int* dims_n, int dim_len, float* data_n, bool copy = true);
    Matrix(int* dims_n, int dim_len, int val);
    ~Matrix();
    Matrix matmul(Matrix other);
    Matrix clone();
    void scmul(float s);
    void add(Matrix& a);
    void subtract(Matrix& a);
    void transpose(int* axes);
    float get(initializer_list<int> pos) const;
    float get_index(int i) const;
    void set(initializer_list<int> pos, float val);
    void set_index(int i, float val);
    void broadcast(int* dim, int dim_len);
    void reshape(int* dims_new, int dim_len_new);
    int* get_broadcasted_strides(int* dims_new, int dim_len_new) const;
    int get_dims_index(int i) const;
    int get_dim_len() const;
    float* get_data() const;
    void print_dims() const;
    static void set_CUDA(bool c);
    static bool get_CUDA();
    
};

extern "C" void matmul_cuda(const float* A, const float* B, float* C, int n, int m, int k);

#endif