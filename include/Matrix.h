#ifndef MATRIX_H
#define MATRIX_H
#include <stdexcept>
#include <iostream>
#include <thread>
#include <arm_neon.h>
#include <random>
#include <algorithm>

class Matrix {

private:

    //flattened approach
    float* data;
    int* dims;
    int* dists;
    int dim_len;
    int data_len;
    size_t aligned_data_len;
    static bool cuda;
    static int tile;
    
    Matrix(int* dims_n, int dim_len, float* data_n, bool copy);//Create a new matrix with given data, can choose to copy or take ownership
    Matrix(int* dims_n, int dim_len, float* data_n, int data_len, int*dists); //Strictly for direct cloning, use incase view has changed (reshape/broadcast).
    int convert_idx(const std::initializer_list<int>& pos) const;//Convert given index to 1d flattend index using strides

    void matmul_cuda(const float* A, const float* B, float* C, int n, int m, int k) const;
    void matmul_cpu_batched(const float* A, const float* B, float* C, const int* this_dists, const int* other_dists, int n, int m, int k, int z) const;
    void matmul_cpu(const float* A, const float* B, float* C, int n, int m, int k) const;

    //Will transpose a matrix physically using simd operations, used internally for efficient matmul and public facing transpose2d.
    void simd_transpose(const float* A, float* B, int n, int m, int z = 0, const int* dists_new = nullptr) const;
    int* get_dims_clone() const;
    int* get_dists_clone() const;
    int* get_broadcasted_strides(const int* dims_new, int dim_len_new) const;//Get the broadcasted strides for this given a set of dimensions
    void set_dim_len(int dim_len_n); // UNCHECKED
    void set_dims(int* dims_n); // UNCHECKED
    void set_dists(int* dists_n); // UNCHECKED
    int* get_dims() const;
    int* get_dists() const;
    float* get_data() const;

    void print_array(const float* arr, int len, int max) const; // Print a given float array
    void print_array(const int* arr, int len, int max) const; // Print a given int array
    
public:
    
    Matrix(const int* dims_n, int dim_len, const float* data_n);//Create a new matrix with a given data array (flattened, row major), will copy data
    Matrix(const int* dims_n, int dim_len, float val);//Create a new matrix filled with a given float
    Matrix(const int* dims_n, int dim_len, unsigned int random_seed = 0);//Create a new matrix filled with random floats between [0-1), can set seed
    ~Matrix();
    Matrix matmul(const Matrix &other) const;//Matmul, extensive docs in source and usage guides. (does not affect original)
    Matrix clone() const;//Return a deep copy clone of this object (does not affect original)
    Matrix scmul(float s);//Will multiply matrix by a scalar,  and return new matrix. (does not affect original)
    Matrix add(const Matrix &other);// Will add two matrices,  and return new matrix. (does not affect original)
    Matrix apply(float (*func)(float));//Will apply a given function, and return new matrix. (does not affect original)
    Matrix transpose2d();//Will transpose data physically leveraging simd, and return new tranposed matrix. (does not affect original)
    void scmul_inplace(float s);//Will multiply matrix by a scalar inplace.
    void add_inplace(const Matrix &other);// Will add two matrices inplace.
    void apply_inplace(float (*func)(float)); // Will apply a given function inplace.
    void transpose_shallow(int* axes); // Will only transpose semantically, will not transpose the data.
    float get(const std::initializer_list<int>& pos) const;//Get a value using format {x, y, z, ...}
    float get_index(int i) const;//Get a value using a flattened index (mostly internal use)
    void set(const std::initializer_list<int>& pos, float val);//Set a value using format {x, y, z, ...}
    void set_index(int i, float val);//Set a value using a flattened index (mostly internal use)
    void broadcast(const int* dim, int dim_len);//Broadcast dimensions, more info in source.
    void reshape(const int* dims_new, int dim_len_new);//Reshape dimensions, more info in source.
    int get_dims_index(int i) const;//Get dims using an flattened index
    int get_dim_len() const;//Get length of all dimensions (returns 3 for 3d)
    void print_dims(int max = 50) const;//Print dims
    void print_dists(int max = 50) const;//Print dists
    void print_data(int max = 50) const;//Print data values up to a given max, default 50
    static void set_CUDA(bool c);//Set CUDA usage
    static bool get_CUDA();//Get CUDA usage
    static void set_tile(int t);
    static int get_tile();

};

extern "C" void matmul_cuda(const float* A, const float* B, float* C, int n, int m, int k);

#endif