#ifndef test_matrix
#define test_matrix
#include <stdexcept>
#include <iostream>
#include <thread>
#include <arm_neon.h>
#include <random>
#include <algorithm>
#include <tuple>

class Matrix {

private:

    //flattened approach
    uint32_t dim_len;
    uint32_t data_len;
    uint32_t aligned_data_len;
    int* dims;
    int* dists;
    float* data;
    static bool cuda;
    static uint16_t tile; // MUST be a multiple of 4.
    static constexpr uint16_t alignment = 32; //16 minimum for SIMD optimization, can change to that

    template <typename T>
    static inline T* assume_aligned(T* ptr) {
        return static_cast<T*>(__builtin_assume_aligned(ptr, alignment));
    }
    
    Matrix(int* dims_n, int dim_len, float* data_n, bool copy);//Create a new matrix with given data, can choose to copy or take ownership
    Matrix(int* dims_n, int dim_len, float* data_n, int data_len, int*dists); //Strictly for direct cloning, use incase view has changed (reshape/broadcast).
    int convert_idx(const std::initializer_list<int>& pos) const;//Convert given index to 1d flattend index using strides

    std::tuple<int,int,int> get_matmul_tile(int n, int m, int k) const; // Get a tile size for a specific matix size. It will assume matrix dimensions are square for fitting into cache for simplicity.
    void matmul_cuda(const float* A, const float* B, float* C, int n, int m, int k) const;
    void matmul_cpu_batched(const float* A, const float* B, float* C, const int* this_dists, const int* other_dists, int n, int m, int k, int z) const;
    void matmul_cpu(const float* A, const float* B, float* C, int n, int m, int k) const;
    void matmul_cpu_naive(const float* A, const float* B, float* C, int n, int m, int k) const; // For comparison.


    //Will transpose a matrix physically using simd operations, used internally for efficient matmul and public facing transpose2d.
    void simd_transpose(const float* A, float* B, int n, int m, int z = 0, const int* dists_new = nullptr) const;
    int* get_dims_clone() const; // Get a deep copy of dims.
    int* get_dists_clone() const; // Get a deep copy of dists array.
    int* get_broadcasted_strides(const int* dims_new, int dim_len_new) const;//Get the broadcasted strides for this given a set of dimensions

    float* default_data_alloc() const; // Allocate an aligned float array based on aligned_data_len.
    float* init_data_alloc(size_t size); // Allocate an aligned float array and initialize aligned_data_len.
    float* float_size_alloc(size_t size) const; //Allocate an aligned float array.
    int* int_size_alloc(size_t size) const; //Allocate an unaligned integer array.

    void print_array(const float* arr, int len, int max) const; // Print a given float array
    void print_array(const int* arr, int len, int max) const; // Print a given int array
    
public:
    
    Matrix(const int* dims_n, int dim_len, const float* data_n); //Create a new matrix with a given data array (flattened, row major), will copy data
    Matrix(const int* dims_n, int dim_len, float val); //Create a new matrix filled with a given float
    Matrix(const int* dims_n, int dim_len, unsigned int random_seed = 0); //Create a new matrix filled with random floats between [0-1), can set seed
    Matrix(const Matrix& other); //Copy constructor
    Matrix& operator=(const Matrix& other); //Copy assignment operator
    Matrix(Matrix&& other) noexcept; //Move constructor
    Matrix& operator=(Matrix&& other) noexcept; //Move assignment operator
    ~Matrix(); //Destructor
    Matrix matmul(const Matrix &other) const; //Matmul, extensive docs in source and usage guides. 
    Matrix clone() const; //Return a deep copy clone of this object
    Matrix scmul(float s) const; //Will multiply matrix by a scalar, and return new matrix.
    Matrix emul(const Matrix &other) const ;// Will multiply two matrices elementwise, and return new matrix. Will prioritize row-col multiplication over col-row multiplication for 1D case
    Matrix add(const Matrix &other) const; // Will add two matrices, and return new matrix. Will prioritize row-col multiplication over col-row multiplication for 1D case
    Matrix apply(float (*func)(float)) const;//Will apply a given function, and return new matrix. 
    Matrix transpose2d() const; // Will transpose data physically leveraging simd, and return new tranposed matrix. 
    void scmul_inplace(float s); // Will multiply matrix by a scalar inplace.
    void emul_inplace(const Matrix &other); // Will multiply two matrices elementwise inplace.
    void add_inplace(const Matrix &other); // Will add two matrices inplace.
    void apply_inplace(float (*func)(float)); // Will apply a given function inplace.
    Matrix sum_rows() const; // Sum all rows into one row vector.
    Matrix sum_cols() const; // Sum all cols into one col vector.
    float sum() const; //Sum all rows and cols into one float.
    float get(const std::initializer_list<int>& pos) const; //Get a value using format {x, y, z, ...}
    float get_index(int i) const; //Get a value using a flattened index (mostly internal use)
    void set(const std::initializer_list<int>& pos, float val); //Set a value using format {x, y, z, ...}
    void set_index(int i, float val); //Set a value using a flattened index (mostly internal use)
    int get_dims_index(int i) const; //Get dims using an flattened index
    int get_dim_len() const; //Get length of all dimensions (returns 3 for 3d)
    void print_dims(int max = 50) const; //Print dims
    void print_dists(int max = 50) const; //Print dists
    void print_data(int max = 50) const; //Print data values up to a given max, default 50
    static void set_CUDA(bool c); //Set CUDA usage
    static bool get_CUDA(); //Get CUDA usage
    static void set_tile(int t);
    static int get_tile();
    static int get_alignment();
};

extern "C" void matmul_cuda(const float* A, const float* B, float* C, int n, int m, int k);

#endif