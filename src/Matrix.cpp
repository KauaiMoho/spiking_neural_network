#include "Matrix.h"
using namespace std;
using std::initializer_list;
#include <stdexcept>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <arm_neon.h>
#include <random>
#include <optional>

//Default do not use CUDA
bool Matrix::cuda = false;

Matrix::Matrix(int* dims_n, int dim_len, float* data_n, bool copy) : dim_len(dim_len), copy(copy) {
    if (dim_len == 0) {
        throw invalid_argument("Matrix dimensions cannot be empty!");
    }

    if (copy) {
        dists = (int*) malloc(dim_len * sizeof(int));

        if (dists == nullptr) {
            throw invalid_argument("Memory allocation error");
        }

        dims = (int*) malloc(dim_len * sizeof(int));

        if (dims == nullptr) {
            throw invalid_argument("Memory allocation error");
        }

        int pos = 1;
        for (int i = dim_len - 1; i > 0 ; i--) {
            dists[i] = pos;
            dims[i] = dims_n[i];
            pos *= dims[i];
        }
        dims[0] = dims_n[0];
        dists[0] = pos;
        data_len = pos*dims[0];


        aligned_data_len = data_len * sizeof(float);
        size_t remainder = aligned_data_len % 16;
        if (remainder != 0) {
            aligned_data_len += 16 - remainder;
        }

        data = (float*) aligned_alloc(16, aligned_data_len);

        if (data == nullptr) {
            throw invalid_argument("Memory allocation error");
        }

        for (size_t i = 0; i < data_len; ++i) {
            data[i] = data_n[i];
        }
    } else {
        dims = dims_n;

        //NEEDS TO BE ALIGNED
        data = data_n;

        dists = (int*) malloc(dim_len * sizeof(int));

        if (dists == nullptr) {
            throw invalid_argument("Memory allocation error");
        }

        int pos = 1;
        for (int i = dim_len - 1; i > 0 ; i--) {
            dists[i] = pos;
            pos *= dims[i];
        }
        dists[0] = pos;
        data_len = pos*dims[0];
    }
}

Matrix::Matrix(int* dims_n, int dim_len, float val) : dim_len(dim_len) {

    if (dim_len == 0) throw invalid_argument("Matrix dimensions cannot be empty!");
    dists = (int*) malloc(dim_len * sizeof(int));

    if (dists == nullptr) {
        throw invalid_argument("Memory allocation error");
    }

    dims = (int*) malloc(dim_len * sizeof(int));

    if (dims == nullptr) {
        throw invalid_argument("Memory allocation error");
    }

    int pos = 1;
    for (int i = dim_len - 1; i > 0 ; i--) {
        dists[i] = pos;
        dims[i] = dims_n[i];
        pos *= dims[i];
    }
    dims[0] = dims_n[0];
    dists[0] = pos;
    data_len = pos*dims[0];

    aligned_data_len = data_len * sizeof(float);
    size_t remainder = aligned_data_len % 16;
    if (remainder != 0) {
        aligned_data_len += 16 - remainder;
    }

    data = (float*) aligned_alloc(16, aligned_data_len);

    if (data == nullptr) {
        throw invalid_argument("Memory allocation error");
    }

    for (size_t i = 0; i < data_len; ++i) {
        data[i] = val;
    }
}

Matrix::Matrix(int* dims_n, int dim_len, unsigned int random_seed) : dim_len(dim_len) {

    if (random_seed == 0) {
        std::random_device rd;
        random_seed = rd();
    }
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::mt19937 gen(random_seed);
    float val = dist(gen);

    if (dim_len == 0) throw invalid_argument("Matrix dimensions cannot be empty!");
    dists = (int*) malloc(dim_len * sizeof(int));

    if (dists == nullptr) {
        throw invalid_argument("Memory allocation error");
    }

    dims = (int*) malloc(dim_len * sizeof(int));

    if (dims == nullptr) {
        throw invalid_argument("Memory allocation error");
    }

    int pos = 1;
    for (int i = dim_len - 1; i > 0 ; i--) {
        dists[i] = pos;
        dims[i] = dims_n[i];
        pos *= dims[i];
    }
    dims[0] = dims_n[0];
    dists[0] = pos;
    data_len = pos*dims[0];

    aligned_data_len = data_len * sizeof(float);
    size_t remainder = aligned_data_len % 16;
    if (remainder != 0) {
        aligned_data_len += 16 - remainder;
    }

    data = (float*) aligned_alloc(16, aligned_data_len);

    if (data == nullptr) {
        throw invalid_argument("Memory allocation error");
    }

    for (size_t i = 0; i < data_len; ++i) {
        data[i] = val;
    }
}

Matrix::~Matrix() {
    //TODO: Bugged
    if (data != nullptr) {
        free(data);
    }
    if (dims != nullptr) {
        free (dims);
    }
    free(dists);
}

static Matrix invalid() {
    return Matrix(nullptr, 0, nullptr, false);
}

int Matrix::convert_idx(initializer_list<int> pos) const {

    //Converts between regular indexing (nd) and stride position (1d)
    int idx = 0;
    int i = 0;
    if (pos.size() != static_cast<size_t>(dim_len)) {
        throw invalid_argument("Wrong number of indices!");
    }
    for (int value : pos) {
        idx += value*dists[i];
        ++i;
    }
    if (idx > data_len - 1 || idx < 0) throw invalid_argument("Invalid position dimensions!");
    return idx;
}

int* Matrix::get_broadcasted_strides(int* dims_new, int dim_len_new) const {
    //Dim_len_new will always be >= dim_len

    //Gets the new strides (row major, flattened) for a given broadcast
    int* dists_new = (int*) malloc(dim_len_new * sizeof(int));
    int diff = dim_len_new - dim_len;
    if (dists_new == nullptr) {
        throw invalid_argument("Memory allocation error");
    }

    for (int i = dim_len_new - 1; i >= 0 ; i--) {
        int i_old = i - diff;
        if (i_old < 0) { 
            dists_new[i] = 0;
        } else {
            if (dims[i_old] == dims_new[i]) {
                dists_new[i] = dists[i_old];
            } else if (dims[i_old] == 1) {
                dists_new[i] = 0;
            } else {
                free(dists_new);
                throw invalid_argument("Incompatible dimensions for broadcasting!");
            }
        }
    }

    return dists_new;
}

void Matrix::reshape(int* dims_new, int dim_len_new) {
    //Dims_new must multiply into same size as dims_old

    //Will change between any dimensions as long as both dimensions have the same total size
    //Data not modified, only stride and dimension array modified
    int dim_size_old = 1;
    for (size_t i = 0; i < dim_len; ++i) {
        dim_size_old *= dims[i];
    }
    int dim_size_new = 1;
    for (size_t i = 0; i < dim_len_new; ++i) {
        dim_size_new *= dims_new[i];
    }
    if (dim_size_old == dim_size_new) {
        free(dists);
        dists = (int*) malloc(dim_len_new * sizeof(int));
        if (dists == nullptr) {
            throw invalid_argument("Memory allocation error");
        }
        free(dims);
        dims = (int*) malloc(dim_len_new * sizeof(int));
        if (dims == nullptr) {
            throw invalid_argument("Memory allocation error");
        }
        int pos = 1;
        for (int i = dim_len_new - 1; i > 0 ; i--) {
            dists[i] = pos;
            dims[i] = dims_new[i];
            pos *= dims[i];
        }
        dims[0] = dims_new[0];
        dists[0] = pos;
        dim_len = dim_len_new;
    } else {    
        throw invalid_argument("Invalid dimension size for reshape!");
    }
}

void Matrix::broadcast(int* dims_new, int dim_len_new) {
    //Dim_len_new must always be >= dim_len

    //Broadcast -> Given two sets of dimensions, will add extra dimensions to the shorter one such that the numbers are extended to match dimensions
    //Useful for a BLAS library when multiplying matrices of unequal dimension
    //Data not modified, only stride and dimension array modified
    if (dim_len_new >= dim_len) {
        int* dists_new = get_broadcasted_strides(dims_new, dim_len_new);
        free(dists);
        free(dims);
        dists = dists_new;
        dims = (int*) malloc(dim_len_new * sizeof(int));
        if (dims == nullptr) {
            throw invalid_argument("Memory allocation error");
        }
        for (size_t i = 0; i < dim_len_new; ++i) {
            dims[i] = dims_new[i];
        }
        dim_len = dim_len_new;
    } else {
        throw invalid_argument("Invalid dimension size for broadcasting!");
    }
}

void Matrix::matmul_cpu_batched(float* A, float* B, float* C, int n, int m, int k, int z) {

    //Use loop order to optimize L Cache loading.
    //Use sysctl -a | grep cache to check Apple Silicon Cache Size

    constexpr size_t L1_bytes = 64 * 1024;
    constexpr size_t L2_bytes = 4 * 1024 * 1024;
    constexpr int cache_line_floats = 32;
    size_t matrix_size_floats = (static_cast<size_t>(n)*m) + (static_cast<size_t>(m)*k) + (static_cast<size_t>(n)*k);

    //Choose between L1 and L2 cache based on matrix size
    size_t usable_cache_bytes;
    if (matrix_size_floats * sizeof(float) <= L1_bytes) {
        usable_cache_bytes = L1_bytes / 1.5;
    } else {
        usable_cache_bytes = L2_bytes / 1.5;
    }

    size_t usable_cache_floats = usable_cache_bytes / sizeof(float);

    //Assumption of near square matrices
    int tile = static_cast<int>(sqrt(usable_cache_floats / 3));
    tile = tile & ~(cache_line_floats - 1);
    if (tile == 0) {
        tile = cache_line_floats;
    }

    // Process small tile of A and corresponding tile of B:
    //     - Load A_tile into cache once
    //     - Load B_tile into cache once
    //     - Compute small block of C_tile

    size_t size = m * k * sizeof(float);
    size_t remainder = size % 16;
    if (remainder != 0) {
        size += 16 - remainder;
    }

    float* B_t = (float*) aligned_alloc(16, size);

    if (B_t == nullptr) {
        throw invalid_argument("Memory allocation error");
    }

    simd_transpose(B, B_t, m, k, z);
    for (int ic = 0; ic < n; ic += tile){
        for (int lc = 0; lc < k; lc += tile){
            int iE = min(ic+tile, n);
            for (size_t i = ic; i < iE; ++i){
                int lE = min(lc+tile, k);
                for (size_t l = lc; l < lE; ++l){
                    float sum = 0;
                    float32x4_t acc = vdupq_n_f32(0.0f);
                    for (int jc = 0; jc < m; jc += tile) {
                        int jE =  min(jc+tile, m);
                        float* ptrA = &A[n*m*z + i*m + jc];
                        float* ptrB = &B_t[l*m + jc];
                        for (size_t j = jc; j + 3 < jE; j += 4) {
                            float32x4_t a = vld1q_f32(ptrA);
                            float32x4_t b = vld1q_f32(ptrB);
                            ptrA += 4;
                            ptrB += 4;
                            acc = vaddq_f32(acc, vmulq_f32(a, b));
                        }
                        for (size_t j = 0; j < (jE%4); ++j) {
                            sum += (*ptrA) * (*ptrB);
                            ptrA += 1;
                            ptrB += 1;
                        }
                    }
                    sum += vaddvq_f32(acc);
                    C[n*k*z + i*k + l] = sum;
                }   
            }
        }
    }
    free(B_t);
}

void Matrix::matmul_cuda(float* A, float* B, float* C, int n, int m, int k) {
    //TODO: Uncomment after compiling with nvcc
    //::matmul_cuda(A, B, C, n, m, k);
}

//A = nxm
//B = mxk
//C = nxk
//Stride A = m
//Stride B = k
//Stride C = k
//Assume matrix dimensions near square for cache optimization simplicity.
void Matrix::matmul_cpu(float* A, float* B, float* C, int n, int m, int k) {

    //Use loop order to optimize L Cache loading.
    //Use sysctl -a | grep cache to check Apple Silicon Cache Size

    constexpr size_t L1_bytes = 64 * 1024;
    constexpr size_t L2_bytes = 4 * 1024 * 1024;
    constexpr int cache_line_floats = 32;
    size_t matrix_size_floats = (static_cast<size_t>(n)*m) + (static_cast<size_t>(m)*k) + (static_cast<size_t>(n)*k);

    //Choose between L1 and L2 cache based on matrix size
    size_t usable_cache_bytes;
    if (matrix_size_floats * sizeof(float) <= L1_bytes) {
        usable_cache_bytes = L1_bytes / 1.5;
    } else {
        usable_cache_bytes = L2_bytes / 1.5;
    }

    size_t usable_cache_floats = usable_cache_bytes / sizeof(float);

    //Assumption of near square matrices, make sure tile is multiple of cache_line_floats
    int tile = static_cast<int>(sqrt(usable_cache_floats / 3));
    tile = tile & ~(cache_line_floats - 1);
    if (tile == 0) {
        tile = cache_line_floats;
    }

    size_t size = m * k * sizeof(float);
    size_t remainder = size % 16;
    if (remainder != 0) {
        size += 16 - remainder;
    }

    float* B_t = (float*) aligned_alloc(16, size);
    
    if (B_t == nullptr) {
        throw invalid_argument("Memory allocation error");
    }
    
    simd_transpose(B, B_t, m, k);
    for (int ic = 0; ic < n; ic += tile){
        for (int lc = 0; lc < k; lc += tile){
            int iE = min(ic+tile, n);
            for (size_t i = ic; i < iE; ++i){
                int lE = min(lc+tile, k);
                for (size_t l = lc; l < lE; ++l){
                    float sum = 0;
                    float32x4_t acc = vdupq_n_f32(0.0f);
                    for (int jc = 0; jc < m; jc += tile) {
                        int jE =  min(jc+tile, m);
                        float* ptrA = &A[i*m + jc];
                        float* ptrB = &B_t[l*m + jc];
                        for (size_t j = jc; j + 3 < jE; j += 4) {
                            float32x4_t a = vld1q_f32(ptrA);
                            float32x4_t b = vld1q_f32(ptrB);
                            ptrA += 4;
                            ptrB += 4;
                            acc = vaddq_f32(acc, vmulq_f32(a, b));
                        }
                        for (size_t j = 0; j < (jE%4); ++j) {
                            sum += (*ptrA) * (*ptrB);
                            ptrA += 1;
                            ptrB += 1;
                        }
                    }
                    sum += vaddvq_f32(acc);
                    C[i*k + l] = sum;
                }   
            }
        }
    }
    free(B_t);
}
void Matrix::simd_transpose(float* A, float* B, int n, int m, int z) {

    size_t tile = 16;
    size_t offset = n*m*z;
    
    //Tile for same reasons as matmul (minimize cache misses)
    for (size_t ic = 0; ic + tile <= n; ic += tile) {
        for (size_t jc = 0; jc + tile <= m; jc += tile) {
            for (size_t i = ic; i < ic+tile; i += 4) {
                for (size_t j = jc; j < jc+tile; j += 4) {
                    //Load 16 elements from A to tranpose into B
                    float32x4_t a = vld1q_f32(&A[(i+0)*m + j + offset]);
                    float32x4_t b = vld1q_f32(&A[(i+1)*m + j + offset]);
                    float32x4_t c = vld1q_f32(&A[(i+2)*m + j + offset]);
                    float32x4_t d = vld1q_f32(&A[(i+3)*m + j + offset]);
                    
                    // a = [a0 a1 a2 a3]
                    // b = [b0 b1 b2 b3]
                    // c = [c0 c1 c2 c3]
                    // d = [d0 d1 d2 d3]

                    //Transpose halves
                    float32x4x2_t p0 = vtrnq_f32(a, b);
                    //[a0 a1 a2 a3]      [a0 b0 a2 b2]
                    //[b0 b1 b2 b3]  →   [a1 b1 a3 b3]
                    float32x4x2_t p1 = vtrnq_f32(c, d);

                    

                    //Combine halves
                    float32x4_t r0 = vcombine_f32(vget_low_f32(p0.val[0]), vget_low_f32(p1.val[0]));
                    
                    // low(p0[0]) = [a0 b0]
                    // low(p1[0]) = [c0 d0]
                    // → r0 = [a0 b0 c0 d0]

                    float32x4_t r1 = vcombine_f32(vget_low_f32(p0.val[1]), vget_low_f32(p1.val[1]));
                    float32x4_t r2 = vcombine_f32(vget_high_f32(p0.val[0]), vget_high_f32(p1.val[0]));

                    // high(p0[0]) = [a2 b2]
                    // high(p1[0]) = [c2 d2]
                    // → r2 = [a2 b2 c2 d2]

                    float32x4_t r3 = vcombine_f32(vget_high_f32(p0.val[1]), vget_high_f32(p1.val[1]));

                    //Store into B
                    vst1q_f32(&B[(j+0)*n + i], r0);
                    vst1q_f32(&B[(j+1)*n + i], r1);
                    vst1q_f32(&B[(j+2)*n + i], r2);
                    vst1q_f32(&B[(j+3)*n + i], r3);
                }
            }
        }
    }


    //Scalar Clean up what was missed by tiling

    //Handles leftover rows top right rectangle and bottom right corner
    for (size_t i = n-(n%tile); i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            B[j*n + i] = A[i*m + j + offset];
        }
    }

    // Corner:
    // i >= n-(n%tile)
    // j >= m-(m%tile)

    //Handle leftover colouns (ignoring final few rows overlapping with above loop)
    //Basically the bottom left rectangle 
    for (size_t i = 0; i < n-(n%tile); ++i) {
        for (size_t j = m-(m%tile); j < m; ++j) {
            B[j*n + i] = A[i*m + j + offset];
        }
    }

}

Matrix Matrix::matmul(Matrix &other) {

    int tile = 16;
    
    if (other.get_dim_len() == 1 && dim_len == 1) {
        //Dimension 1 x 1 = Dot product
        
        if (other.get_dims_index(0) == dims[0]) {
            int* new_dims = (int*) malloc(sizeof(int));
            if (new_dims == nullptr) {
                throw invalid_argument("Memory allocation error");
            }
            new_dims[0] = 1;
            float data_out = 0;
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (int jc = 0; jc < data_len; jc += tile) {
                int jE =  min(jc+tile, data_len);
                const float* oPtr = other.get_data() + jc;
                float* tPtr = data + jc;
                for (size_t j = jc; j + 3 < jE; j += 4) {
                    float32x4_t a = vld1q_f32(oPtr);
                    float32x4_t b = vld1q_f32(tPtr);
                    oPtr += 4;
                    tPtr += 4;
                    acc = vaddq_f32(acc, vmulq_f32(a, b));
                }
                for (size_t j = 0; j < (jE%4); ++j) {
                    data_out += (*oPtr) * (*tPtr);
                    oPtr += 1;
                    tPtr += 1;
                }
            }
            data_out += vaddvq_f32(acc);
            Matrix ret = Matrix(new_dims, 1, data_out);
            free(new_dims);
            return ret;
        }
        throw invalid_argument("Invalid dot product dimensions!");
    } else if (other.get_dim_len() == 1 && dim_len == 2) {
        // dimension 2 x 1 = Vector Product

        //n x m X m x 1 = n x 1
        if (other.get_dims_index(0) == dims[1]) {
            int* new_dims = (int*) malloc(sizeof(int));
            if (new_dims == nullptr) {
                throw invalid_argument("Memory allocation error");
            }
            new_dims[0] = dims[0];

            size_t size = new_dims[0] * sizeof(float);
            size_t remainder = size % 16;
            if (remainder != 0) {
                size += 16 - remainder;
            }

            float* data_out = (float*) aligned_alloc(16, size);

            if (data_out == nullptr) {
                throw invalid_argument("Memory allocation error");
            }

            for (int ic = 0; ic < new_dims[0]; ic += tile){
                int iE = min(ic+tile, new_dims[0]);
                for (size_t i = ic; i < iE; ++i){
                    float sum = 0;
                    float32x4_t acc = vdupq_n_f32(0.0f);
                    for (int jc = 0; jc < dims[1]; jc += tile) {
                        int jE =  min(jc+tile, dims[1]);
                        float* ptrA = &data[i*dims[1] + jc];
                        float* ptrB = &(other.get_data()[jc]);
                        for (size_t j = jc; j + 3 < jE; j += 4) {
                            float32x4_t a = vld1q_f32(ptrA);
                            float32x4_t b = vld1q_f32(ptrB);
                            ptrA += 4;
                            ptrB += 4;
                            acc = vaddq_f32(acc, vmulq_f32(a, b));
                        }
                        for (size_t j = 0; j < (jE%4); ++j) {
                            sum += (*ptrA) * (*ptrB);
                            ptrA += 1;
                            ptrB += 1;
                        }
                    }
                    sum += vaddvq_f32(acc);
                    data_out[i] = sum;
                }   
            }

            Matrix ret = Matrix(new_dims, 1, data_out, false);
            return ret;
        }
        throw invalid_argument("Invalid matrix-vector product dimensions!");
    } else if (other.get_dim_len() == 2 && dim_len == 1) {
        // dimension 1 x 2 = Vector Product
        if (other.get_dims_index(1) == dims[0]) {
            int* new_dims = (int*) malloc(sizeof(int));
            if (new_dims == nullptr) {
                throw invalid_argument("Memory allocation error");
            }
            new_dims[0] = other.get_dims_index(0);

            int other_dim = other.get_dims_index(1);

            size_t size = new_dims[0] * sizeof(float);
            size_t remainder = size % 16;
            if (remainder != 0) {
                size += 16 - remainder;
            }

            float* data_out = (float*) aligned_alloc(16, size);

            if (data_out == nullptr) {
                throw invalid_argument("Memory allocation error");
            }

            for (int ic = 0; ic < new_dims[0]; ic += tile){
                int iE = min(ic+tile, new_dims[0]);
                for (size_t i = ic; i < iE; ++i){
                    float sum = 0;
                    float32x4_t acc = vdupq_n_f32(0.0f);
                    for (int jc = 0; jc < other_dim; jc += tile) {
                        int jE =  min(jc+tile, other_dim);
                        float* ptrA = &(other.get_data()[i*other_dim + jc]);
                        float* ptrB = &data[jc];
                        for (size_t j = jc; j + 3 < jE; j += 4) {
                            float32x4_t a = vld1q_f32(ptrA);
                            float32x4_t b = vld1q_f32(ptrB);
                            ptrA += 4;
                            ptrB += 4;
                            acc = vaddq_f32(acc, vmulq_f32(a, b));
                        }
                        for (size_t j = 0; j < (jE%4); ++j) {
                            sum += (*ptrA) * (*ptrB);
                            ptrA += 1;
                            ptrB += 1;
                        }
                    }
                    sum += vaddvq_f32(acc);
                    data_out[i] = sum;
                }   
            }


            for (int i = 0; i < new_dims[0]; ++i) {
                float sum = 0;
                for (int j = 0; j < dims[0]; ++j) {
                    sum += get({j})*other.get({i, j});
                }
                data_out[i] = sum;
            }
            Matrix ret = Matrix(new_dims, 1, data_out, false);
            return ret;
        }
        throw invalid_argument("Invalid vector-matrix product dimensions!");
    } else if (other.get_dim_len() == 2 && dim_len == 2) {
        // Dimension 2 x 2 = Matrix multiplication
        // Will perform This X Other
        if (dims[1] == other.get_dims_index(0)) {

            int* new_dims = (int*) malloc(2 * sizeof(int));
            if (new_dims == nullptr) {
                throw invalid_argument("Memory allocation error");
            }
            new_dims[0] = dims[0];
            new_dims[1] = other.get_dims_index(1);

            size_t size = new_dims[0] * new_dims[1] * sizeof(float);
            size_t remainder = size % 16;
            if (remainder != 0) {
                size += 16 - remainder;
            }

            float* data_out = (float*) aligned_alloc(16, size);

            if (data_out == nullptr) {
                throw invalid_argument("Memory allocation error");
            } 
            if (cuda) {
                matmul_cuda(data, other.get_data(), data_out, new_dims[0], dims[1], new_dims[1]);
            } else {
                matmul_cpu(data, other.get_data(), data_out, new_dims[0], dims[1], new_dims[1]);
            }
            Matrix ret = Matrix(new_dims, 2, data_out, false);
            return ret;
        }
        throw invalid_argument("Invalid matrix-matrix product dimensions!");
    } else if (other.get_dim_len() >= 2 && dim_len >= 2) {
        // Dimension n x n = Batched matrix multiplaction with broadcasting
        //Will perform This X Other, batched
        int other_dim_len = other.get_dim_len();
        if (dims[dim_len-1] == other.get_dims_index(other_dim_len - 2)) {
            int broadcast_dim_len = max(dim_len, other_dim_len);
            int* broadcast_dims = (int*) malloc(broadcast_dim_len * sizeof(int));
            if (broadcast_dims == nullptr) {
                throw invalid_argument("Memory allocation error");
            }
            if (dim_len >= other_dim_len) {
                int diff = broadcast_dim_len - other_dim_len;
                for (int i = dim_len - 3; i >= 0 ; --i) {
                    int i_other = i - diff;
                    if (i_other < 0) { 
                        broadcast_dims[i] = dims[i];
                    } else {
                        int other_dim = other.get_dims_index(i_other);
                        if (dims[i] == other_dim || other_dim == 1) {
                            broadcast_dims[i] = dims[i];
                        } else if (dims[i] == 1) {
                            broadcast_dims[i] = other_dim;
                        } else {
                            free(broadcast_dims);
                            throw invalid_argument("Incompatible dimensions for matmul batch broadcasting!");
                        }
                    }
                }
            } else {
                int diff = broadcast_dim_len - dim_len;
                for (int i = other_dim_len - 3; i >= 0 ; --i) {
                    int other_dim = other.get_dims_index(i);
                    int i_this = i - diff;
                    if (i_this < 0) { 
                        broadcast_dims[i] = other_dim;
                    } else {
                        if (dims[i_this] == other_dim || other_dim == 1) {
                            broadcast_dims[i] = dims[i_this];
                        } else if (dims[i_this] == 1) {
                            broadcast_dims[i] = other_dim;
                        } else {
                            free(broadcast_dims);
                            throw invalid_argument("Incompatible dimensions for matmul batch broadcasting!");
                        }
                    }
                }
            }
            int* bmm_shape = (int*) malloc(3 * sizeof(int));
            if (bmm_shape == nullptr) {
                throw invalid_argument("Memory allocation error");
            }
            bmm_shape[0] = 1;
            for (size_t i = 0; i < broadcast_dim_len - 2; ++i) {
                bmm_shape[0] *= broadcast_dims[i];
            }

            //Save old dimensions, do not free
            int dim_len_this = dim_len;
            int dim_len_other = other.get_dim_len();
            int* dims_clone_this = get_dims_clone();
            int* dims_clone_other = other.get_dims_clone();
            int* dists_clone_this = get_dists_clone();
            int* dists_clone_other = other.get_dists_clone();
            
            //Broadcast and reshape
            broadcast_dims[broadcast_dim_len - 2] = dims[dim_len - 2];
            broadcast_dims[broadcast_dim_len - 1] = dims[dim_len - 1];
            broadcast(broadcast_dims, broadcast_dim_len);

            broadcast_dims[broadcast_dim_len - 2] = other.get_dims_index(other_dim_len - 2);
            broadcast_dims[broadcast_dim_len - 1] = other.get_dims_index(other_dim_len - 1);
            other.broadcast(broadcast_dims, broadcast_dim_len);
            free(broadcast_dims);
            
            bmm_shape[1] = dims[broadcast_dim_len - 2];
            bmm_shape[2] = dims[broadcast_dim_len - 1];
            reshape(bmm_shape, 3);

            bmm_shape[1] = other.get_dims_index(broadcast_dim_len - 2);
            bmm_shape[2] = other.get_dims_index(broadcast_dim_len - 1);
            other.reshape(bmm_shape, 3);
            bmm_shape[1] = dims[1];

            //matmul
            int n_threads = thread::hardware_concurrency();

            //Avoid malloc to call constructor
            thread* threads = new thread[n_threads];

            size_t size = bmm_shape[0] * dims[dim_len - 2] * bmm_shape[2] * sizeof(float);
            size_t remainder = size % 16;
            if (remainder != 0) {
                size += 16 - remainder;
            }

            float* data_out = (float*) aligned_alloc(16, size);

            if (data_out == nullptr) {
                throw invalid_argument("Memory allocation error");
            } 
            
            //each thread handles an set of indivisual slices (divided evenly between all possible threads)
            for (size_t t = 0; t < n_threads; ++t) {
                threads[t] = thread([&, t]() {
                    for (size_t i = t; i < bmm_shape[0]; i += n_threads) {
                        matmul_cpu_batched(data, other.get_data(), data_out, dims[dim_len - 2],
                                        dims[dim_len - 1], bmm_shape[2], i);
                    }
                });
            }
            for (size_t i = 0; i < n_threads; ++i) {
                threads[i].join();
            }

            Matrix ret = Matrix(bmm_shape, 3, data_out, false);

            delete[] threads;

            dim_len = dim_len_this;
            dims = dims_clone_this;
            dists = dists_clone_this;
            other.set_dim_len(dim_len_other);
            other.set_dims(dims_clone_other);
            other.set_dists(dists_clone_other);

            return ret;
        } else {
            throw invalid_argument("Invalid batched matrix-matrix product dimensions!");
        }
    }
    cout << "INVALID";
    return invalid();
}   

Matrix Matrix::clone() const {
    return Matrix(dims, dim_len, data);
}

Matrix Matrix::scmul(float s) {

    float* data_out = (float*) aligned_alloc(16, aligned_data_len);

    if (data_out == nullptr) {
        throw invalid_argument("Memory allocation error");
    } 

    float32x4_t scalar = vdupq_n_f32(s);
    float* tPtr = data;
    float* outPtr = data_out;

    for (size_t i = 0; i + 3 < data_len; i += 4) {
        float32x4_t t = vld1q_f32(tPtr);
        float32x4_t res = vmulq_f32(t, scalar);
        vst1q_f32(outPtr, res);
        tPtr += 4;
        outPtr += 4;
    }

    for (size_t i = 0; i < data_len % 4; ++i) {
        outPtr[i] = tPtr[i] * s;
    }

    Matrix ret = Matrix(get_dims_clone(), dim_len, data_out, false);
    return ret;
}


Matrix Matrix::add(const Matrix& other) {

    for (size_t i = 0; i < dim_len; ++i){
        if (dims[i] != other.get_dims_index(i)) {
            throw invalid_argument("Invalid matrix dimensions!");
        }
    }

    float* data_out = (float*) aligned_alloc(16, aligned_data_len);

    if (data_out == nullptr) {
        throw invalid_argument("Memory allocation error");
    }
    
    const float* oPtr = other.get_data();
    float* tPtr = data;
    float* outPtr = data_out;

    for (size_t i = 0; i + 3 < data_len; i += 4) {

        float32x4_t af = vld1q_f32(oPtr);
        float32x4_t tf = vld1q_f32(tPtr);
        float32x4_t add = vaddq_f32(af, tf);
        vst1q_f32(outPtr, add);
        oPtr += 4;
        tPtr += 4;
        outPtr += 4;
    }

    for (size_t i = 0; i < data_len % 4; ++i) {
        outPtr[i] = tPtr[i] + oPtr[i];
    }

    Matrix ret = Matrix(get_dims_clone(), dim_len, data_out, false);
    return ret;
}

Matrix Matrix::subtract(const Matrix& other) {

    for (size_t i = 0; i < dim_len; ++i){
        if (dims[i] != other.get_dims_index(i)) {
            throw invalid_argument("Invalid matrix dimensions!");
        }
    }

    float* data_out = (float*) aligned_alloc(16, aligned_data_len);

    if (data_out == nullptr) {
        throw invalid_argument("Memory allocation error");
    }
    
    const float* oPtr = other.get_data();
    float* tPtr = data;
    float* outPtr = data_out;

    for (size_t i = 0; i + 3 < data_len; i += 4) {

        float32x4_t af = vld1q_f32(oPtr);
        float32x4_t tf = vld1q_f32(tPtr);
        float32x4_t sub = vsubq_f32(tf, af);
        vst1q_f32(outPtr, sub);
        oPtr += 4;
        tPtr += 4;
        outPtr += 4;
    }

    for (size_t i = 0; i < data_len % 4; ++i) {
        outPtr[i] = tPtr[i] - oPtr[i];
    }

    Matrix ret = Matrix(get_dims_clone(), dim_len, data_out, false);
    return ret;
}

Matrix Matrix::apply(float (*func)(float)) {

    float* data_out = (float*) aligned_alloc(16, aligned_data_len);

    if (data_out == nullptr) {
        throw invalid_argument("Memory allocation error");
    }
    
    for (size_t i = 0; i < data_len; ++i) {
        data_out[i] = func(data[i]);
    }

    Matrix ret = Matrix(get_dims_clone(), dim_len, data_out, false);
    return ret;
}

void Matrix::scmul_inplace(float s) {

    float32x4_t scalar = vdupq_n_f32(s);
    float* tPtr = data;

    for (size_t i = 0; i + 3 < data_len; i += 4) {
        float32x4_t t = vld1q_f32(tPtr);
        float32x4_t res = vmulq_f32(t, scalar);
        vst1q_f32(tPtr, res);
        tPtr += 4;
    }

    for (size_t i = 0; i < data_len % 4; ++i) {
        tPtr[i] = tPtr[i] * s;
    }
}

void Matrix::add_inplace(const Matrix& other) {

    for (size_t i = 0; i < dim_len; ++i){
        if (dims[i] != other.get_dims_index(i)) {
            throw invalid_argument("Invalid matrix dimensions!");
        }
    }
    
    const float* oPtr = other.get_data();
    float* tPtr = data;

    for (size_t i = 0; i + 3 < data_len; i += 4) {

        float32x4_t af = vld1q_f32(oPtr);
        float32x4_t tf = vld1q_f32(tPtr);
        float32x4_t add = vaddq_f32(af, tf);
        vst1q_f32(tPtr, add);
        oPtr += 4;
        tPtr += 4;
    }

    for (size_t i = 0; i < data_len % 4; ++i) {
        tPtr[i] = tPtr[i] + oPtr[i];
    }
}

void Matrix::subtract_inplace(const Matrix& other) {
    for (size_t i = 0; i < dim_len; ++i){
        if (dims[i] != other.get_dims_index(i)) {
            throw invalid_argument("Invalid matrix dimensions!");
        }
    }
    
    const float* oPtr = other.get_data();
    float* tPtr = data;

    for (size_t i = 0; i + 3 < data_len; i += 4) {

        float32x4_t af = vld1q_f32(oPtr);
        float32x4_t tf = vld1q_f32(tPtr);
        float32x4_t sub = vsubq_f32(tf, af);
        vst1q_f32(tPtr, sub);
        oPtr += 4;
        tPtr += 4;
    }

    for (size_t i = 0; i < data_len % 4; ++i) {
        tPtr[i] = tPtr[i] - oPtr[i];
    }
}

void Matrix::apply_inplace(float (*func)(float)) {
    for (size_t i = 0; i < data_len; ++i) {
        data[i] = func(data[i]);
    }
}

void Matrix::transpose(int* axes) {
    int* dists_c = (int*) malloc(dim_len * sizeof(int));
    if (dists_c == nullptr) {
        throw invalid_argument("Memory allocation error");
    }
    for (size_t i = 0; i < dim_len; ++i) {
        dists_c[i] = dists[axes[i]];
    }
    for (size_t i = 0; i < dim_len; ++i) {
        dists[i] = dists_c[i];
    }
    free(dists_c);
}

float Matrix::get(const initializer_list<int> &pos) const {
    return data[convert_idx(pos)];
}

float Matrix::get_index(int i) const {
    if (i < 0 || i > data_len-1){
        throw invalid_argument("Invalid index!");
    }
    return data[i];
}

void Matrix::set(const initializer_list<int> &pos, float val) {
    data[convert_idx(pos)] = val;
}

void Matrix::set_index(int i, float val) {
    if (i < 0 || i > data_len-1) {
        throw invalid_argument("Invalid index!");
    }
    data[i] = val;
}

int Matrix::get_dims_index(int i) const {
    if (i < 0 || i > dim_len-1) {
        throw invalid_argument("Invalid index!");
    }
    return dims[i];
}

int Matrix::get_dim_len() const {
    return dim_len;
}

int* Matrix::get_dists_clone() const {
    int* dists_clone = (int*) malloc(dim_len * sizeof(int));
    if (dists_clone == nullptr) {
        throw invalid_argument("Memory allocation error");
    }
    for (size_t i = 0; i < dim_len; ++i) {
        dists_clone[i] = dists[i];
    }
    return dists_clone;
}

int* Matrix::get_dims_clone() const {
    int* dims_clone = (int*) malloc(dim_len * sizeof(int));
    if (dims_clone == nullptr) {
        throw invalid_argument("Memory allocation error");
    }
    for (size_t i = 0; i < dim_len; ++i) {
        dims_clone[i] = dims[i];
    }
    return dims_clone;
}

void Matrix::set_dim_len(int dim_len_n) {
    //UNCHECKED, INTERNAL BMM USE ONLY
    dim_len = dim_len_n;
}

void Matrix::set_dims(int* dims_n) {
    //UNCHECKED, INTERNAL BMM USE ONLY
    dims = dims_n;
}

void Matrix::set_dists(int* dists_n) {
    //UNCHECKED, INTERNAL BMM USE ONLY
    dists = dists_n;
}

float* Matrix::get_data() const {
    return data;
}

void Matrix::print_data(int m) const {
    int end = min(data_len, m);
    for (size_t i = 0; i < end; ++i) {
        cout << data[i];
        if (i < end - 1) {
             cout << ", ";
        }
    }
    cout << "\n";
}

void Matrix::print_dims() const {
    for (size_t i = 0; i < dim_len; ++i) {
        cout << dims[i];
        cout << " ";
    }
    cout << "\n";
}


//Get/Set CUDA Usage
void Matrix::set_CUDA(bool c) {
    cuda = c;
}

bool Matrix::get_CUDA() {
    return cuda;
}