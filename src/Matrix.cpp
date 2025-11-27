#include "Matrix.h"
using namespace std;
using std::initializer_list;
#include <stdexcept>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <arm_neon.h>

//TODO
// CHANGE MAIN VECTOR OPERATIONS TO USE SIMD - Single Instruction, Multiple Data

//Default do not use CUDA
bool Matrix::cuda = false;

Matrix::Matrix(int* dims_n, int dim_len, float* data_n) : dim_len(dim_len) {
    //Define the dists var here
    //Assuming data/dims will be freed outside of constructor.
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


    size_t size = data_len * sizeof(float);
    size_t remainder = size % 16;
    if (remainder != 0) {
        size += 16 - remainder;
    }

    data = (float*) aligned_alloc(16, size);

    if (data == nullptr) {
        throw invalid_argument("Memory allocation error");
    }

    for (int i = 0; i < data_len; ++i) {
        data[i] = data_n[i];
    }
}

Matrix::Matrix(int* dims_n, int dim_len, int val) : dim_len(dim_len) {

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

    size_t size = data_len * sizeof(float);
    size_t remainder = size % 16;
    if (remainder != 0) {
        size += 16 - remainder;
    }

    data = (float*) aligned_alloc(16, size);

    if (data == nullptr) {
        throw invalid_argument("Memory allocation error");
    }

    for (int i = 0; i < data_len; ++i) {
        data[i] = val;
    }
}

static Matrix invalid() {
    return Matrix(0, 0, nullptr);
}

int Matrix::convert_idx(initializer_list<int> pos) const {
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
    int dim_size_old = 1;
    for (int i = 0; i < dim_len; ++i) {
        dim_size_old *= dims[i];
    }
    int dim_size_new = 1;
    for (int i = 0; i < dim_len_new; ++i) {
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
    if (dim_len_new >= dim_len) {
        int* dists_new = get_broadcasted_strides(dims_new, dim_len_new);
        free(dists);
        free(dims);
        dists = dists_new;
        dims = (int*) malloc(dim_len_new * sizeof(int));
        if (dims == nullptr) {
            throw invalid_argument("Memory allocation error");
        }
        for (int i = 0; i < dim_len_new; ++i) {
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

    simd_transpose(B, B_t, m, k);
    for (int ic = 0; ic < n; ic += tile){
        for (int lc = 0; lc < k; lc += tile){
            int iE = min(ic+tile, n);
            for (int i = ic; i < iE; ++i){
                int lE = min(lc+tile, k);
                for (int l = lc; l < lE; ++l){
                    float sum = 0;
                    float32x4_t acc = vdupq_n_f32(0.0f);
                    for (int jc = 0; jc < m; jc += tile) {
                        int jE =  min(jc+tile, m);
                        float* ptrA = &A[n*m*z + i*m + jc];
                        float* ptrB = &B_t[n*m*z + l*m + jc];
                        for (int j = jc; j + 3 < jE; j += 4) {
                            float32x4_t a = vld1q_f32(ptrA);
                            float32x4_t b = vld1q_f32(ptrB);
                            ptrA += 4;
                            ptrB += 4;
                            acc = vaddq_f32(acc, vmulq_f32(a, b));
                        }
                        for (int j = jE - (jE%4); j < jE; ++j) {
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
            for (int i = ic; i < iE; ++i){
                int lE = min(lc+tile, k);
                for (int l = lc; l < lE; ++l){
                    float sum = 0;
                    float32x4_t acc = vdupq_n_f32(0.0f);
                    for (int jc = 0; jc < m; jc += tile) {
                        int jE =  min(jc+tile, m);
                        float* ptrA = &A[i*m + jc];
                        float* ptrB = &B_t[l*m + jc];
                        for (int j = jc; j + 3 < jE; j += 4) {
                            float32x4_t a = vld1q_f32(ptrA);
                            float32x4_t b = vld1q_f32(ptrB);
                            ptrA += 4;
                            ptrB += 4;
                            acc = vaddq_f32(acc, vmulq_f32(a, b));
                        }
                        for (int j = jE - (jE%4); j < jE; ++j) {
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
void Matrix::simd_transpose(float* A, float* B, int n, int m) {

    int tile = 16;
    
    for (int ic = 0; ic + tile <= n; ic += tile) {
        for (int jc = 0; jc + tile <= m; jc += tile) {
            for (int i = ic; i < ic+tile; i += 4) {
                for (int j = jc; j < jc+tile; j += 4) {
                    //Load 16 elements from A to tranpose into B
                    float32x4_t a = vld1q_f32(&A[(i+0)*m + j]);
                    float32x4_t b = vld1q_f32(&A[(i+1)*m + j]);
                    float32x4_t c = vld1q_f32(&A[(i+2)*m + j]);
                    float32x4_t d = vld1q_f32(&A[(i+3)*m + j]);

                    //Transpose halves
                    float32x4x2_t p0 = vtrnq_f32(a, b);
                    float32x4x2_t p1 = vtrnq_f32(c, d);

                    //Combine halves
                    float32x4_t r0 = vcombine_f32(vget_low_f32(p0.val[0]), vget_low_f32(p1.val[0]));
                    float32x4_t r1 = vcombine_f32(vget_low_f32(p0.val[1]), vget_low_f32(p1.val[1]));
                    float32x4_t r2 = vcombine_f32(vget_high_f32(p0.val[0]), vget_high_f32(p1.val[0]));
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

    for (int i = n-(n%tile); i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            B[j*n + i] = A[i*m + j];
        }
    }
    for (int i = 0; i < n-(n%tile); ++i) {
        for (int j = m-(m%tile); j < m; ++j) {
            B[j*n + i] = A[i*m + j];
        }
    }
}

Matrix Matrix::matmul(Matrix other) {
    
    if (other.get_dim_len() == 1 && dim_len == 1) {
        //Dimension 1 x 1 = Dot product
        if (other.get_dims_index(0) == dims[0]) {
            int* new_dims = (int*) malloc(sizeof(int));
            if (new_dims == nullptr) {
                throw invalid_argument("Memory allocation error");
            }
            new_dims[0] = 1;
            float data_out = 0;
            float* data_temp = other.get_data();
            for (int i = 0; i < data_len; ++i) {
                data_out += data[i]*data_temp[i];
            }
            Matrix ret = Matrix(new_dims, 1, data_out);
            free(new_dims);
            return ret;
        }
        throw invalid_argument("Invalid dot product dimensions!");
    } else if (other.get_dim_len() == 1 && dim_len == 2) {
        // dimension 2 x 1 = Vector Product
        if (other.get_dims_index(0) == dims[1]) {
            int* new_dims = (int*) malloc(sizeof(int));
            if (new_dims == nullptr) {
                throw invalid_argument("Memory allocation error");
            }
            new_dims[0] = dims[0];
            float* data_out = (float*) malloc(new_dims[0] * sizeof(float));
            if (data_out == nullptr) {
                throw invalid_argument("Memory allocation error");
            }
            for (int i = 0; i < new_dims[0]; ++i) {
                float sum = 0;
                for (int j = 0; j < dims[1]; ++j) {
                    sum += get({i, j})*other.get({j});
                }
                data_out[i] = sum;
            }
            Matrix ret = Matrix(new_dims, 1, data_out);
            free(new_dims);
            free(data_out);
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
            float* data_out = (float*) malloc(new_dims[0] * sizeof(float));
            if (data_out == nullptr) {
                throw invalid_argument("Memory allocation error");
            }
            for (int i = 0; i < new_dims[0]; ++i) {
                float sum = 0;
                for (int j = 0; j < dims[0]; ++j) {
                    sum += get({j})*other.get({i, j});
                }
                data_out[i] = sum;
            }
            Matrix ret = Matrix(new_dims, 1, data_out);
            free(new_dims);
            free(data_out);
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
            float* data_out = (float*) malloc(new_dims[0] * new_dims[1] * sizeof(float));
            if (data_out == nullptr) {
                throw invalid_argument("Memory allocation error");
            } 
            if (cuda) {
                matmul_cuda(data, other.get_data(), data_out, new_dims[0], dims[1], new_dims[1]);
            } else {
                matmul_cpu(data, other.get_data(), data_out, new_dims[0], dims[1], new_dims[1]);
            }
            Matrix ret = Matrix(new_dims, 2, data_out);
            free(new_dims);
            free(data_out);
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
                for (int i = dim_len - 3; i >= 0 ; i--) {
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
                for (int i = other_dim_len - 3; i >= 0 ; i--) {
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
            for (int i = 0; i < broadcast_dim_len - 2; ++i) {
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
            thread* threads = new thread[bmm_shape[0]];

            float* data_out = (float*) malloc(bmm_shape[0] * dims[dim_len - 2] * bmm_shape[2] * sizeof(float));
            if (data_out == nullptr) {
                throw invalid_argument("Memory allocation error");
            } 

            for (int t = 0; t < n_threads; ++t) {
                threads[t] = thread([&, t]() {
                    for (int i = t; i < bmm_shape[0]; i += n_threads) {
                        matmul_cpu_batched(data, other.get_data(), data_out, dims[dim_len - 2],
                                        dims[dim_len - 1], bmm_shape[2], i);
                    }
                });
            }
            for (int i = 0; i < n_threads; ++i) {
                threads[i].join();
            }

            delete[] threads;

            Matrix ret = Matrix(bmm_shape, 3, data_out);
            free(data_out);
            free(bmm_shape);

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
    return invalid();
}   

Matrix Matrix::clone() {
    return Matrix(dims, dim_len, data);
}

void Matrix::scmul(float s) {

    float32x4_t scalar = vdupq_n_f32(s);
    float* tPtr = &data[0];

    for (int i = 0; i + 3 < data_len; i += 4) {
        float32x4_t t = vld1q_f32(tPtr);
        float32x4_t res = vmulq_f32(t, scalar);
        vst1q_f32(tPtr, res);
        tPtr += 4;
    }

    for (int i = data_len - (data_len % 4); i < data_len; ++i) {
        data[i] = data[i] * s;
    }
}

void Matrix::add(Matrix& a) {
    for (int i = 0; i < dim_len; ++i){
        if (dims[i] != a.get_dims_index(i)) {
            throw invalid_argument("Invalid matrix dimensions!");
        }
    }
    
    float* aPtr = &a.get_data()[0];
    float* tPtr = &data[0];

    for (int i = 0; i + 3 < data_len; i += 4) {

        float32x4_t a = vld1q_f32(aPtr);
        float32x4_t t = vld1q_f32(tPtr);
        float32x4_t add = vaddq_f32(a, t);
        vst1q_f32(tPtr, add);
        aPtr += 4;
        tPtr += 4;
    }

    for (int i = data_len - (data_len % 4); i < data_len; ++i) {
        data[i] = data[i] + a.get_index(i);
    }
}

void Matrix::subtract(Matrix& a) {
    for (int i = 0; i < dim_len; ++i){
        if (dims[i] != a.get_dims_index(i)) {
            throw invalid_argument("Invalid matrix dimensions!");
        }
    }
    
    float* aPtr = &a.get_data()[0];
    float* tPtr = &data[0];

    for (int i = 0; i + 3 < data_len; i += 4) {

        float32x4_t a = vld1q_f32(aPtr);
        float32x4_t t = vld1q_f32(tPtr);
        float32x4_t sub = vsubq_f32(t, a);
        vst1q_f32(tPtr, sub);
        aPtr += 4;
        tPtr += 4;
    }

    for (int i = data_len - (data_len % 4); i < data_len; ++i) {
        data[i] = data[i] - a.get_index(i);
    }
}

void Matrix::transpose(int* axes) {
    int* dists_c = (int*) malloc(dim_len * sizeof(int));
    if (dists_c == nullptr) {
        throw invalid_argument("Memory allocation error");
    }
    for (int i = 0; i < dim_len; ++i) {
        dists_c[i] = dists[axes[i]];
    }
    for (int i = 0; i < dim_len; ++i) {
        dists[i] = dists_c[i];
    }
    free(dists_c);
}

float Matrix::get(initializer_list<int> pos) const {
    return data[convert_idx(pos)];
}

float Matrix::get_index(int i) const {
    if (i < 0 || i > data_len-1){
        throw invalid_argument("Invalid index!");
    }
    return data[i];
}

void Matrix::set(initializer_list<int> pos, float val) {
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

int* Matrix::get_dists_clone() {
    int* dists_clone = (int*) malloc(dim_len * sizeof(int));
    if (dists_clone == nullptr) {
        throw invalid_argument("Memory allocation error");
    }
    for (int i = 0; i < dim_len; ++i) {
        dists_clone[i] = dists[i];
    }
    return dists_clone;
}

int* Matrix::get_dims_clone() {
    int* dims_clone = (int*) malloc(dim_len * sizeof(int));
    if (dims_clone == nullptr) {
        throw invalid_argument("Memory allocation error");
    }
    for (int i = 0; i < dim_len; ++i) {
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

void Matrix::print_dims() const {
    for (int i = 0; i < dim_len; ++i) {
        cout << dims[i];
        cout << " ";
    }
    cout << "\n";
}

void Matrix::set_CUDA(bool c) {
    cuda = c;
}

bool Matrix::get_CUDA() {
    return cuda;
}