#include "Matrix.h"

//Default do not use CUDA
bool Matrix::cuda = false;
int Matrix::tile = 16;

Matrix::Matrix(const int* dims_n, int dim_len, const float* data_n) : dim_len(dim_len) {
    if (dim_len == 0) {
        throw std::invalid_argument("Matrix dimensions cannot be empty!");
    }

    dists = (int*) malloc(dim_len * sizeof(int));

    if (dists == nullptr) {
        throw std::invalid_argument("Memory allocation error");
    }

    dims = (int*) malloc(dim_len * sizeof(int));

    if (dims == nullptr) {
        throw std::invalid_argument("Memory allocation error");
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
        throw std::invalid_argument("Memory allocation error");
    }

    for (size_t i = 0; i < data_len; ++i) {
        data[i] = data_n[i];
    }
}

Matrix::Matrix(int* dims_n, int dim_len, float* data_n, int data_len, int*dists_n) : dim_len(dim_len), data_len(data_len) {
    if (dim_len == 0) {
        throw std::invalid_argument("Matrix dimensions cannot be empty!");
    }

    //Will copy dists as well incase of broadcasting
    dists = (int*) malloc(dim_len * sizeof(int));

    if (dists == nullptr) {
        throw std::invalid_argument("Memory allocation error");
    }

    dims = (int*) malloc(dim_len * sizeof(int));

    if (dims == nullptr) {
        throw std::invalid_argument("Memory allocation error");
    }

    for (int i = 0; i < dim_len ; ++i) {
        dists[i] = dists_n[i];
        dims[i] = dims_n[i];
    }

    aligned_data_len = data_len * sizeof(float);
    size_t remainder = aligned_data_len % 16;
    if (remainder != 0) {
        aligned_data_len += 16 - remainder;
    }

    data = (float*) aligned_alloc(16, aligned_data_len);

    if (data == nullptr) {
        throw std::invalid_argument("Memory allocation error");
    }

    for (size_t i = 0; i < data_len; ++i) {
        data[i] = data_n[i];
    }
}

Matrix::Matrix(int* dims_n, int dim_len, float* data_n, bool copy) : dim_len(dim_len) {
    if (dim_len == 0) {
        throw std::invalid_argument("Matrix dimensions cannot be empty!");
    }

    if (copy) {
        //Same as public constructor - can change later to get of repeated code
        dists = (int*) malloc(dim_len * sizeof(int));

        if (dists == nullptr) {
            throw std::invalid_argument("Memory allocation error");
        }

        dims = (int*) malloc(dim_len * sizeof(int));

        if (dims == nullptr) {
            throw std::invalid_argument("Memory allocation error");
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
            throw std::invalid_argument("Memory allocation error");
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
            throw std::invalid_argument("Memory allocation error");
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

Matrix::Matrix(const int* dims_n, int dim_len, float val) : dim_len(dim_len) {

    if (dim_len == 0) {
        throw std::invalid_argument("Matrix dimensions cannot be empty!");
    }

    dists = (int*) malloc(dim_len * sizeof(int));

    if (dists == nullptr) {
        throw std::invalid_argument("Memory allocation error");
    }

    dims = (int*) malloc(dim_len * sizeof(int));

    if (dims == nullptr) {
        throw std::invalid_argument("Memory allocation error");
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
        throw std::invalid_argument("Memory allocation error");
    }

    for (size_t i = 0; i < data_len; ++i) {
        data[i] = val;
    }
}

Matrix::Matrix(const int* dims_n, int dim_len, unsigned int random_seed) : dim_len(dim_len) {

    if (dim_len == 0) {
        throw std::invalid_argument("Matrix dimensions cannot be empty!");
    }

    if (random_seed == 0) {
        std::random_device rd;
        random_seed = rd();
    }
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::mt19937 gen(random_seed);
    float val = dist(gen);

    dists = (int*) malloc(dim_len * sizeof(int));

    if (dists == nullptr) {
        throw std::invalid_argument("Memory allocation error");
    }

    dims = (int*) malloc(dim_len * sizeof(int));

    if (dims == nullptr) {
        throw std::invalid_argument("Memory allocation error");
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
        throw std::invalid_argument("Memory allocation error");
    }

    for (size_t i = 0; i < data_len; ++i) {
        data[i] = val;
    }
}

Matrix::~Matrix() {
    //NOTE, CANNOT USE = operator: Matrix A = B
    //TODO: check
    free(data);
    free(dims);
    free(dists);
}

void Matrix::print_array(const float* arr, int len, int max) const {
    int end = std::min(len, max);
    for (size_t i = 0; i < end; ++i) {
        std::cout << arr[i];
        if (i < end - 1) {
             std::cout << ", ";
        }
    }
    std::cout << "\n";
}

void Matrix::print_array(const int* arr, int len, int max) const {
    int end = std::min(len, max);
    for (size_t i = 0; i < end; ++i) {
        std::cout << arr[i];
        if (i < end - 1) {
             std::cout << ", ";
        }
    }
    std::cout << "\n";
}

int Matrix::convert_idx(const std::initializer_list<int>& pos) const {

    //Converts between regular indexing (nd) and stride position (1d)
    int idx = 0;
    int i = 0;
    if (pos.size() != static_cast<size_t>(dim_len)) {
        throw std::invalid_argument("Wrong number of indices!");
    }
    for (int value : pos) {
        idx += value*dists[i];
        ++i;
    }
    if (idx > data_len - 1 || idx < 0) throw std::invalid_argument("Invalid position dimensions!");
    return idx;
}

int* Matrix::get_broadcasted_strides(const int* dims_new, int dim_len_new) const {
    //Gets the new strides (row major, flattened) for a given broadcast
    if (dim_len_new >= dim_len) {
        int* dists_new = (int*) malloc(dim_len_new * sizeof(int));
        int diff = dim_len_new - dim_len;
        if (dists_new == nullptr) {
            throw std::invalid_argument("Memory allocation error");
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
                    throw std::invalid_argument("Incompatible dimensions for broadcasting!");
                }
            }
        }
        return dists_new;
    } else {
        throw std::invalid_argument("Invalid dimension size for broadcasting!");
    }
}

void Matrix::matmul_cpu_batched(const float* A, const float* B, float* C, const int* this_dists, const int* other_dists, int n, int m, int k, int z) const {

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
        throw std::invalid_argument("Memory allocation error");
    }

    simd_transpose(B, B_t, m, k, z, other_dists);

    for (int ic = 0; ic < n; ic += tile){
        for (int lc = 0; lc < k; lc += tile){
            int iE = std::min(ic+tile, n);
            for (size_t i = ic; i < iE; ++i){
                int lE = std::min(lc+tile, k);
                for (size_t l = lc; l < lE; ++l){
                    float sum = 0;
                    float32x4_t acc = vdupq_n_f32(0.0f);
                    for (int jc = 0; jc < m; jc += tile) {
                        int jE =  std::min(jc+tile, m);

                        //Broadcasted strides will have a dimension of 0 in strides, allowing for still efficient cache usage
                        const float* ptrA = &A[z*this_dists[0] + i*this_dists[1] + jc*this_dists[2]];
                        float* ptrB = &B_t[l*m + jc];
                        for (size_t j = jc; j + 3 < jE; j += 4) {
                            float32x4_t a = vld1q_f32(ptrA);
                            float32x4_t b = vld1q_f32(ptrB);
                            ptrA += 4 * this_dists[2];
                            ptrB += 4;
                            acc = vaddq_f32(acc, vmulq_f32(a, b));
                        }
                        for (size_t j = 0; j < (jE%4); ++j) {
                            sum += (*ptrA) * (*ptrB);
                            ptrA += this_dists[2];
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

void Matrix::matmul_cuda(const float* A, const float* B, float* C, int n, int m, int k) const {
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
void Matrix::matmul_cpu(const float* A, const float* B, float* C, int n, int m, int k) const {

    //Use loop order to optimize L Cache loading.
    //Use sysctl -a | grep cache to check Apple Silicon Cache Size

    constexpr size_t L1_bytes = 64 * 1024;
    constexpr size_t L2_bytes = 4 * 1024 * 1024;
    constexpr int cache_line_floats = 16;
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
        throw std::invalid_argument("Memory allocation error");
    }
    
    //We choose to transpose the data physically because reading col - major is inefficient for cache, even if we just transpose the 
    //dist strides the cache reading will be slow for large matrices +, wanted to challenge myself to write a simd transpose.
    simd_transpose(B, B_t, m, k);
    for (int ic = 0; ic < n; ic += tile){
        for (int lc = 0; lc < k; lc += tile){
            int iE = std::min(ic+tile, n);
            for (size_t i = ic; i < iE; ++i){
                int lE = std::min(lc+tile, k);
                for (size_t l = lc; l < lE; ++l){
                    float sum = 0;
                    float32x4_t acc = vdupq_n_f32(0.0f);
                    for (int jc = 0; jc < m; jc += tile) {
                        int jE =  std::min(jc+tile, m);
                        const float* ptrA = &A[i*m + jc];
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

void Matrix::simd_transpose(const float* A, float* B, int n, int m, int z, const int* dists_new) const {
    
    size_t tile = 16;

    //We choose to repeat code rather than make a temp dists var so that the original dists_new can stay const (cannot free if 
    //we use const temp_dists_new).

    if (!dists_new) {
        size_t offset = n*m*z;
        //Tile for same reasons as matmul (std::minimize cache misses)
        for (size_t ic = 0; ic + tile <= n; ic += tile) {
            for (size_t jc = 0; jc + tile <= m; jc += tile) {
                for (size_t i = ic; i < ic+tile; i += 4) {
                    for (size_t j = jc; j < jc+tile; j += 4) {
                        //Load 16 elements from A to tranpose into B
                        // a = [a0 a1 a2 a3]
                        // b = [b0 b1 b2 b3]
                        // c = [c0 c1 c2 c3]
                        // d = [d0 d1 d2 d3]
                        float32x4_t a = vld1q_f32(&A[offset + i*m + j]);
                        float32x4_t b = vld1q_f32(&A[offset + (i + 1)*m + j]);
                        float32x4_t c = vld1q_f32(&A[offset + (i + 2)*m + j]);
                        float32x4_t d = vld1q_f32(&A[offset + (i + 3)*m + j]);

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
                        vst1q_f32(&B[j*n + i], r0);
                        vst1q_f32(&B[(j + 1)*n + i], r1);
                        vst1q_f32(&B[(j + 2)*n + i], r2);
                        vst1q_f32(&B[(j + 3)*n + i], r3);
                    }
                }
            }
        }

        //Scalar Clean up what was missed by tiling

        //Handles leftover rows - Bottom Rectangle
        for (size_t i = n-(n%tile); i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                B[j*n + i] = A[offset + i*m + j];
            }
        }

        // Corner:
        // i >= n-(n%tile)
        // j >= m-(m%tile)

        //Handle leftover colouns (ignoring final few rows overlapping with above loop)
        //Basically the top right rectangle 
        for (size_t i = 0; i < n-(n%tile); ++i) {
            for (size_t j = m-(m%tile); j < m; ++j) {
                B[j*n + i] = A[offset + i*m + j];
            }
        }

    } else {
        
        for (size_t ic = 0; ic + tile <= n; ic += tile) {
            for (size_t jc = 0; jc + tile <= m; jc += tile) {
                for (size_t i = ic; i < ic+tile; i += 4) {
                    for (size_t j = jc; j < jc+tile; j += 4) {
                        float32x4_t a = vld1q_f32(&A[z*dists_new[0] + i*dists_new[1] + j*dists_new[2]]);
                        float32x4_t b = vld1q_f32(&A[z*dists_new[0] + (i + 1)*dists_new[1] + j*dists_new[2]]);
                        float32x4_t c = vld1q_f32(&A[z*dists_new[0] + (i + 2)*dists_new[1] + j*dists_new[2]]);
                        float32x4_t d = vld1q_f32(&A[z*dists_new[0] + (i + 3)*dists_new[1] + j*dists_new[2]]);
                        
                        float32x4x2_t p0 = vtrnq_f32(a, b);
                        float32x4x2_t p1 = vtrnq_f32(c, d);

                        float32x4_t r0 = vcombine_f32(vget_low_f32(p0.val[0]), vget_low_f32(p1.val[0]));
                        float32x4_t r1 = vcombine_f32(vget_low_f32(p0.val[1]), vget_low_f32(p1.val[1]));
                        float32x4_t r2 = vcombine_f32(vget_high_f32(p0.val[0]), vget_high_f32(p1.val[0]));
                        float32x4_t r3 = vcombine_f32(vget_high_f32(p0.val[1]), vget_high_f32(p1.val[1]));

                        vst1q_f32(&B[j*n + i], r0);
                        vst1q_f32(&B[(j + 1)*n + i], r1);
                        vst1q_f32(&B[(j + 2)*n + i], r2);
                        vst1q_f32(&B[(j + 3)*n + i], r3);
                    }
                }
            }
        }

        for (size_t i = n-(n%tile); i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                B[j*n + i] = A[z*dists_new[0] + i*dists_new[1] + j*dists_new[2]];
            }
        }

        for (size_t i = 0; i < n-(n%tile); ++i) {
            for (size_t j = m-(m%tile); j < m; ++j) {
                B[j*n + i] = A[z*dists_new[0] + i*dists_new[1] + j*dists_new[2]];
            }
        }
    }
}

Matrix Matrix::matmul(const Matrix& other) const {
    
    if (other.get_dim_len() == 1 && dim_len == 1) {
        //Dimension 1 x 1 = Dot product
        
        if (other.get_dims_index(0) == dims[0]) {
            int* new_dims = (int*) malloc(sizeof(int));
            if (new_dims == nullptr) {
                throw std::invalid_argument("Memory allocation error");
            }
            new_dims[0] = 1;
            float data_out = 0;
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (int jc = 0; jc < data_len; jc += tile) {
                int jE =  std::min(jc+tile, data_len);
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
        throw std::invalid_argument("Invalid dot product dimensions!");
    } else if (other.get_dim_len() == 1 && dim_len == 2) {
        // dimension 2 x 1 = Vector Product

        //n x m X m x 1 = n x 1
        if (other.get_dims_index(0) == dims[1]) {
            int* new_dims = (int*) malloc(sizeof(int));
            if (new_dims == nullptr) {
                throw std::invalid_argument("Memory allocation error");
            }
            new_dims[0] = dims[0];

            size_t size = new_dims[0] * sizeof(float);
            size_t remainder = size % 16;
            if (remainder != 0) {
                size += 16 - remainder;
            }

            float* data_out = (float*) aligned_alloc(16, size);

            if (data_out == nullptr) {
                throw std::invalid_argument("Memory allocation error");
            }

            for (int ic = 0; ic < new_dims[0]; ic += tile){
                int iE = std::min(ic+tile, new_dims[0]);
                for (size_t i = ic; i < iE; ++i){
                    float sum = 0;
                    float32x4_t acc = vdupq_n_f32(0.0f);
                    for (int jc = 0; jc < dims[1]; jc += tile) {
                        int jE =  std::min(jc+tile, dims[1]);
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
        throw std::invalid_argument("Invalid matrix-vector product dimensions!");
    } else if (other.get_dim_len() == 2 && dim_len == 1) {
        // dimension 1 x 2 = Vector Product
        //1 x m X m x k = 1 x k
        if (other.get_dims_index(0) == dims[0]) {
            int* new_dims = (int*) malloc(sizeof(int));
            if (new_dims == nullptr) {
                throw std::invalid_argument("Memory allocation error");
            }

            //Transpose to avoid col-wide inefficent access

            size_t m = other.get_dims_index(0);
            size_t k = other.get_dims_index(1);

            size_t size_other = m * k * sizeof(float);
            size_t remainder_other = size_other % 16;
            if (remainder_other != 0) {
                size_other += 16 - remainder_other;
            }

            float* other_t = (float*) aligned_alloc(16, size_other);
            
            if (other_t == nullptr) {
                throw std::invalid_argument("Memory allocation error");
            }

            float* data_other = other.get_data();
            simd_transpose(data_other, other_t, m, k);

            new_dims[0] = k;

            size_t size = new_dims[0] * sizeof(float);
            size_t remainder = size % 16;
            if (remainder != 0) {
                size += 16 - remainder;
            }

            float* data_out = (float*) aligned_alloc(16, size);

            if (data_out == nullptr) {
                throw std::invalid_argument("Memory allocation error");
            }

            for (int ic = 0; ic < new_dims[0]; ic += tile){
                int iE = std::min(ic+tile, new_dims[0]);
                for (size_t i = ic; i < iE; ++i){
                    float sum = 0;
                    float32x4_t acc = vdupq_n_f32(0.0f);
                    for (int jc = 0; jc < dims[0]; jc += tile) {
                        int jE =  std::min(jc+tile, dims[0]);
                        float* ptrA = &data[jc];
                        float* ptrB = &other_t[i*dims[0] + jc];
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

            free(other_t);
            Matrix ret = Matrix(new_dims, 1, data_out, false);
            return ret;
        }
        throw std::invalid_argument("Invalid vector-matrix product dimensions!");
    } else if (other.get_dim_len() == 2 && dim_len == 2) {
        // Dimension 2 x 2 = Matrix multiplication
        // Will perform This X Other
        if (dims[1] == other.get_dims_index(0)) {

            int* new_dims = (int*) malloc(2 * sizeof(int));
            if (new_dims == nullptr) {
                throw std::invalid_argument("Memory allocation error");
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
                throw std::invalid_argument("Memory allocation error");
            } 
            if (cuda) {
                matmul_cuda(data, other.get_data(), data_out, new_dims[0], dims[1], new_dims[1]);
            } else {
                matmul_cpu(data, other.get_data(), data_out, new_dims[0], dims[1], new_dims[1]);
            }
            Matrix ret = Matrix(new_dims, 2, data_out, false);
            return ret;
        }
        throw std::invalid_argument("Invalid matrix-matrix product dimensions!");
    } else if (other.get_dim_len() >= 2 && dim_len >= 2) {
        // Dimension n x n = Batched matrix multiplaction with broadcasting
        //Will perform This X Other, batched
        int other_dim_len = other.get_dim_len();
        if (dims[dim_len-1] == other.get_dims_index(other_dim_len - 2)) {
            int broadcast_dim_len = std::max(dim_len, other_dim_len);
            int* broadcast_dims = (int*) malloc(broadcast_dim_len * sizeof(int));
            if (broadcast_dims == nullptr) {
                throw std::invalid_argument("Memory allocation error");
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
                            throw std::invalid_argument("Incompatible dimensions for matmul batch broadcasting!");
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
                            throw std::invalid_argument("Incompatible dimensions for matmul batch broadcasting!");
                        }
                    }
                }
            }

            //We MUST allocate this on the heap due to how the destructor for this class works - free will fail.
            // - Could add a boolean flag to notify destructor if stack-allocated, but increased complexity/less readable
            int* bmm_shape = (int*) malloc(3 * sizeof(int));

            if (bmm_shape == nullptr) {
                throw std::invalid_argument("Memory allocation error");
            }

            bmm_shape[0] = 1;
            bmm_shape[1] = dims[dim_len - 2];
            bmm_shape[2] = other.get_dims_index(other_dim_len - 1);
            for (size_t i = 0; i < broadcast_dim_len - 2; ++i) {
                bmm_shape[0] *= broadcast_dims[i];
            }
            
            //Broadcast: preserve last two dimensions for matmul.
            broadcast_dims[broadcast_dim_len - 2] = dims[dim_len - 2];
            broadcast_dims[broadcast_dim_len - 1] = dims[dim_len - 1];
            int* this_dists = get_broadcasted_strides(broadcast_dims, broadcast_dim_len);

            broadcast_dims[broadcast_dim_len - 2] = other.get_dims_index(other_dim_len - 2);
            broadcast_dims[broadcast_dim_len - 1] = other.get_dims_index(other_dim_len - 1);
            int* other_dists = other.get_broadcasted_strides(broadcast_dims, broadcast_dim_len);

            free(broadcast_dims);

            int n_threads = std::thread::hardware_concurrency();

            //Avoid malloc to call constructor
            std::thread* threads = new std::thread[n_threads];

            size_t size = bmm_shape[0] * dims[dim_len - 2] * bmm_shape[2] * sizeof(float);
            size_t remainder = size % 16;
            if (remainder != 0) {
                size += 16 - remainder;
            }

            float* data_out = (float*) aligned_alloc(16, size);

            if (data_out == nullptr) {
                throw std::invalid_argument("Memory allocation error");
            }

            //each thread handles an set of indivisual slices (divided evenly between all possible threads)
            for (size_t t = 0; t < n_threads; ++t) {
                threads[t] = std::thread([&, t]() {
                    for (size_t i = t; i < bmm_shape[0]; i += n_threads) {
                        matmul_cpu_batched(data, other.get_data(), data_out, this_dists, other_dists, 
                                dims[dim_len - 2], dims[dim_len - 1], bmm_shape[2], i);
                    }
                });
            }
            for (size_t i = 0; i < n_threads; ++i) {
                threads[i].join();
            }
            
            delete[] threads;

            free(this_dists);
            free(other_dists);

            Matrix ret = Matrix(bmm_shape, 3, data_out, false);
            return ret;

        } else {
            throw std::invalid_argument("Invalid batched matrix-matrix product dimensions!");
        }
    }
    return Matrix(nullptr, 0, nullptr, false);
}   

Matrix Matrix::clone() const {
    return Matrix(dims, dim_len, data, data_len, dists);
}

Matrix Matrix::scmul(float s) const {

    float* data_out = (float*) aligned_alloc(16, aligned_data_len);

    if (data_out == nullptr) {
        throw std::invalid_argument("Memory allocation error");
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

//Will prioritize row-col addition over col-row addition
Matrix Matrix::emul(const Matrix& other) const {

    //Matrix - Vector element mul - specific, quicker kernel for ANN
    //Could generalize in future by doing broadcasting, but would be largely the same as matmul
    //Must be 2D and row major (cannot be semantically reshaped/broadcast/transpose)
    if (other.get_dim_len() == 1 && dim_len == 2) {

        if (other.get_dims_index(0) == dims[1]) { //Add row to cols

            float* data_out = (float*) aligned_alloc(16, aligned_data_len);
            if (data_out == nullptr) {
                throw std::invalid_argument("Memory allocation error");
            }

            float* tPtr = data;
            float* outPtr = data_out;
            float* oPtr = other.get_data();

            //Dont do tiling to match rest of add, can do in future.
            for (size_t i = 0; i < dims[0]; ++i){
                for (size_t j = 0; j + 3 < dims[1]; j += 4) {
                    float32x4_t tf = vld1q_f32(tPtr);
                    float32x4_t af = vld1q_f32(&oPtr[j]);
                    float32x4_t mul = vmulq_f32(af, tf);
                    vst1q_f32(outPtr, mul);
                    tPtr += 4;
                    outPtr += 4;
                }
                for (size_t j = 0; j < (dims[1]%4); ++j) {
                    (*outPtr) = (*tPtr) * oPtr[j];
                    outPtr += 1;
                    tPtr += 1;
                }
            }

            Matrix ret = Matrix(get_dims_clone(), dim_len, data_out, false);
            return ret;

        } else if (other.get_dims_index(0) == dims[0]) { //Add col to rows

            float* data_out = (float*) aligned_alloc(16, aligned_data_len);
            if (data_out == nullptr) {
                throw std::invalid_argument("Memory allocation error");
            }

            float* tPtr = data;
            float* outPtr = data_out;
            float* oPtr = other.get_data();

            //Dont do tiling to match rest of add, can do in future.
            for (size_t i = 0; i < dims[0]; ++i){
                float vec_float = *oPtr;
                float32x4_t af = vdupq_n_f32(vec_float);
                for (size_t j = 0; j + 3 < dims[1]; j += 4) {
                    float32x4_t tf = vld1q_f32(tPtr);
                    float32x4_t mul = vmulq_f32(af, tf);
                    vst1q_f32(outPtr, mul);
                    tPtr += 4;
                    outPtr += 4;
                }
                for (size_t j = 0; j < (dims[1]%4); ++j) {
                    (*outPtr) = (*tPtr) * vec_float;
                    tPtr += 1;
                    outPtr += 1;
                }
                oPtr += 1;   
            }

            Matrix ret = Matrix(get_dims_clone(), dim_len, data_out, false);
            return ret;

        } else {
            throw std::invalid_argument("Invalid matrix-vector add dimensions!");
        }

    } else { //General

        for (size_t i = 0; i < dim_len; ++i){
            if (dims[i] != other.get_dims_index(i)) {
                throw std::invalid_argument("Invalid matrix-matrix add dimensions!");
            }
        }

        float* data_out = (float*) aligned_alloc(16, aligned_data_len);

        if (data_out == nullptr) {
            throw std::invalid_argument("Memory allocation error");
        }
        
        const float* oPtr = other.get_data();
        float* tPtr = data;
        float* outPtr = data_out;

        for (size_t i = 0; i + 3 < data_len; i += 4) {

            float32x4_t af = vld1q_f32(oPtr);
            float32x4_t tf = vld1q_f32(tPtr);
            float32x4_t mul = vmulq_f32(af, tf);
            vst1q_f32(outPtr, mul);
            oPtr += 4;
            tPtr += 4;
            outPtr += 4;
        }

        for (size_t i = 0; i < data_len % 4; ++i) {
            outPtr[i] = tPtr[i] * oPtr[i];
        }

        Matrix ret = Matrix(get_dims_clone(), dim_len, data_out, false);
        return ret;

    }
}

//Will prioritize row-col addition over col-row addition
Matrix Matrix::add(const Matrix& other) const {

    //Matrix - Vector add - specific, quicker kernel for ANN
    //Could generalize in future by doing broadcasting, but would be largely the same as matmul
    //Must be 2D and row major (cannot be semantically reshaped/broadcast/transpose)
    if (other.get_dim_len() == 1 && dim_len == 2) {

        if (other.get_dims_index(0) == dims[1]) { //Add row to cols

            float* data_out = (float*) aligned_alloc(16, aligned_data_len);
            if (data_out == nullptr) {
                throw std::invalid_argument("Memory allocation error");
            }

            float* tPtr = data;
            float* outPtr = data_out;
            float* oPtr = other.get_data();

            //Dont do tiling to match rest of add, can do in future.
            for (size_t i = 0; i < dims[0]; ++i){
                for (size_t j = 0; j + 3 < dims[1]; j += 4) {
                    float32x4_t tf = vld1q_f32(tPtr);
                    float32x4_t af = vld1q_f32(&oPtr[j]);
                    float32x4_t add = vaddq_f32(af, tf);
                    vst1q_f32(outPtr, add);
                    tPtr += 4;
                    outPtr += 4;
                }
                for (size_t j = 0; j < (dims[1]%4); ++j) {
                    (*outPtr) = (*tPtr) + oPtr[j];
                    outPtr += 1;
                    tPtr += 1;
                }
            }

            Matrix ret = Matrix(get_dims_clone(), dim_len, data_out, false);
            return ret;

        } else if (other.get_dims_index(0) == dims[0]) { //Add col to rows

            float* data_out = (float*) aligned_alloc(16, aligned_data_len);
            if (data_out == nullptr) {
                throw std::invalid_argument("Memory allocation error");
            }

            float* tPtr = data;
            float* outPtr = data_out;
            float* oPtr = other.get_data();

            //Dont do tiling to match rest of add, can do in future.
            for (size_t i = 0; i < dims[0]; ++i){
                float vec_float = *oPtr;
                float32x4_t af = vdupq_n_f32(vec_float);
                for (size_t j = 0; j + 3 < dims[1]; j += 4) {
                    float32x4_t tf = vld1q_f32(tPtr);
                    float32x4_t add = vaddq_f32(af, tf);
                    vst1q_f32(outPtr, add);
                    tPtr += 4;
                    outPtr += 4;
                }
                for (size_t j = 0; j < (dims[1]%4); ++j) {
                    (*outPtr) = (*tPtr) + vec_float;
                    tPtr += 1;
                    outPtr += 1;
                }
                oPtr += 1;   
            }

            Matrix ret = Matrix(get_dims_clone(), dim_len, data_out, false);
            return ret;

        } else {
            throw std::invalid_argument("Invalid matrix-vector add dimensions!");
        }

    } else { //General Tensor Add

        for (size_t i = 0; i < dim_len; ++i){
            if (dims[i] != other.get_dims_index(i)) {
                throw std::invalid_argument("Invalid matrix-matrix add dimensions!");
            }
        }

        float* data_out = (float*) aligned_alloc(16, aligned_data_len);

        if (data_out == nullptr) {
            throw std::invalid_argument("Memory allocation error");
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
}

Matrix Matrix::apply(float (*func)(float)) const {

    float* data_out = (float*) aligned_alloc(16, aligned_data_len);

    if (data_out == nullptr) {
        throw std::invalid_argument("Memory allocation error");
    }
    
    for (size_t i = 0; i < data_len; ++i) {
        data_out[i] = func(data[i]);
    }

    Matrix ret = Matrix(get_dims_clone(), dim_len, data_out, false);
    return ret;
}

Matrix Matrix::transpose2d() const {
    if (dim_len == 2) {

        float* data_out = (float*) aligned_alloc(16, aligned_data_len);

        if (data_out == nullptr) {
            throw std::invalid_argument("Memory allocation error");
        }

        simd_transpose(data, data_out, dims[0], dims[1]);

        int* dims_new = (int*) malloc(dim_len * sizeof(int));

        if (dims_new == nullptr) {
            throw std::invalid_argument("Memory allocation error");
        }

        dims_new[0] = dims[1];
        dims_new[1] = dims[0];

        Matrix ret = Matrix(dims_new, dim_len, data_out, false);
        return ret;
    } else {
        throw std::invalid_argument("Invalid matrix dimensions! Must be 2d");
    }
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

void Matrix::emul_inplace(const Matrix& other) {

    if (other.get_dim_len() == 1 && dim_len == 2) {

        if (other.get_dims_index(0) == dims[1]) {

            float* tPtr = data;
            float* oPtr = other.get_data();

            for (size_t i = 0; i < dims[0]; ++i){
                for (size_t j = 0; j + 3 < dims[1]; j += 4) {
                    float32x4_t tf = vld1q_f32(tPtr);
                    float32x4_t af = vld1q_f32(&oPtr[j]);
                    float32x4_t mul = vmulq_f32(af, tf);
                    vst1q_f32(tPtr, mul);
                    tPtr += 4;
                }
                for (size_t j = 0; j < (dims[1]%4); ++j) {
                    (*tPtr) = (*tPtr) * oPtr[j];
                    tPtr += 1;
                }
            }

        } else if (other.get_dims_index(0) == dims[0]) {

            float* tPtr = data;
            float* oPtr = other.get_data();

            for (int i = 0; i < dims[0]; ++i){
                float vec_float = *oPtr;
                float32x4_t af = vdupq_n_f32(vec_float);
                for (size_t j = 0; j + 3 < dims[1]; j += 4) {
                    float32x4_t tf = vld1q_f32(tPtr);
                    float32x4_t mul = vmulq_f32(af, tf);
                    vst1q_f32(tPtr, mul);
                    tPtr += 4;
                }
                for (size_t j = 0; j < (dims[1]%4); ++j) {
                    (*tPtr) = (*tPtr) * vec_float;
                    tPtr += 1;
                }
                oPtr += 1;   
            }

        } else {
            throw std::invalid_argument("Invalid matrix-vector add dimensions!");
        }

    } else {

        for (size_t i = 0; i < dim_len; ++i){
            if (dims[i] != other.get_dims_index(i)) {
                throw std::invalid_argument("Invalid matrix dimensions!");
            }
        }
        
        const float* oPtr = other.get_data();
        float* tPtr = data;

        for (size_t i = 0; i + 3 < data_len; i += 4) {

            float32x4_t af = vld1q_f32(oPtr);
            float32x4_t tf = vld1q_f32(tPtr);
            float32x4_t mul = vmulq_f32(af, tf);
            vst1q_f32(tPtr, mul);
            oPtr += 4;
            tPtr += 4;
        }

        for (size_t i = 0; i < data_len % 4; ++i) {
            tPtr[i] = tPtr[i] * oPtr[i];
        }
    }
}

//Will prioritize row-col addition over col-row addition
void Matrix::add_inplace(const Matrix& other) {

    if (other.get_dim_len() == 1 && dim_len == 2) {

        if (other.get_dims_index(0) == dims[1]) { //Add row to cols

            float* tPtr = data;
            float* oPtr = other.get_data();

            //Dont do tiling to match rest of add, can do in future.
            for (size_t i = 0; i < dims[0]; ++i){
                for (size_t j = 0; j + 3 < dims[1]; j += 4) {
                    float32x4_t tf = vld1q_f32(tPtr);
                    float32x4_t af = vld1q_f32(&oPtr[j]);
                    float32x4_t add = vaddq_f32(af, tf);
                    vst1q_f32(tPtr, add);
                    tPtr += 4;
                }
                for (size_t j = 0; j < (dims[1]%4); ++j) {
                    (*tPtr) = (*tPtr) + oPtr[j];
                    tPtr += 1;
                }
            }

        } else if (other.get_dims_index(0) == dims[0]) { //Add col to rows

            float* tPtr = data;
            float* oPtr = other.get_data();

            //Dont do tiling to match rest of add, can do in future.
            for (int i = 0; i < dims[0]; ++i){
                float vec_float = *oPtr;
                float32x4_t af = vdupq_n_f32(vec_float);
                for (size_t j = 0; j + 3 < dims[1]; j += 4) {
                    float32x4_t tf = vld1q_f32(tPtr);
                    float32x4_t add = vaddq_f32(af, tf);
                    vst1q_f32(tPtr, add);
                    tPtr += 4;
                }
                for (size_t j = 0; j < (dims[1]%4); ++j) {
                    (*tPtr) = (*tPtr) + vec_float;
                    tPtr += 1;
                }
                oPtr += 1;   
            }

        } else {
            throw std::invalid_argument("Invalid matrix-vector add dimensions!");
        }

    } else {

        for (size_t i = 0; i < dim_len; ++i){
            if (dims[i] != other.get_dims_index(i)) {
                throw std::invalid_argument("Invalid matrix dimensions!");
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
}

void Matrix::apply_inplace(float (*func)(float)) {
    for (size_t i = 0; i < data_len; ++i) {
        data[i] = func(data[i]);
    }
}

//Must be 2D and row major. Returns a 1D matrix with sum of all rows.
Matrix Matrix::sum_rows() const {
    if (dim_len == 2) {

        int* new_dims = (int*) malloc(sizeof(int));
        if (new_dims == nullptr) {
            throw std::invalid_argument("Memory allocation error");
        }
        new_dims[0] = dims[1];

        size_t size = dims[1] * sizeof(float);

        size_t remainder = size % 16;
        if (remainder != 0) {
            size += 16 - remainder;
        }

        float* data_out = (float*) aligned_alloc(16, size);

        if (data_out == nullptr) {
            throw std::invalid_argument("Memory allocation error");
        }

        float* tPtr = data;
        float* outPtr = data_out;

        for (size_t j = 0; j + 3 < dims[1]; ++j) {
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (size_t i = 0; i < dims[0]; ++i) {
                float32x4_t tf = vld1q_f32(tPtr + i*dims[1] + j);
                acc = vaddq_f32(acc, tf);
            }
            vst1q_f32(outPtr, acc);
            outPtr += 4;
        } 

        for (size_t j = 0; j < dims[1]%4; ++j) {
            (*outPtr) = 0;
            for (size_t i = 0; i < dims[0]; ++i) {
                (*outPtr) += tPtr[i*dims[1] + j];
            }
            outPtr += 1;
        }

        Matrix ret = Matrix(new_dims, 1, data_out, false);
        return ret;

    } else {
        throw std::invalid_argument("Invalid dimensions, must be 2D");
    }
}

//Must be 2D and row major. Returns a 1D matrix with sum of all cols.
Matrix Matrix::sum_cols() const {
    if (dim_len == 2) {

        int* new_dims = (int*) malloc(sizeof(int));
        if (new_dims == nullptr) {
            throw std::invalid_argument("Memory allocation error");
        }
        new_dims[0] = dims[0];

        size_t size = dims[0] * sizeof(float);

        size_t remainder = size % 16;
        if (remainder != 0) {
            size += 16 - remainder;
        }

        float* data_out = (float*) aligned_alloc(16, size);

        if (data_out == nullptr) {
            throw std::invalid_argument("Memory allocation error");
        }

        float* tPtr = data;
        float* outPtr = data_out;

        for (size_t i = 0; i < dims[0]; ++i) {
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (size_t j = 0; j + 3 < dims[1]; j += 4) {
                float32x4_t tf = vld1q_f32(tPtr);
                acc = vaddq_f32(acc, tf);
                tPtr += 4;
            }
            (*outPtr) = vaddvq_f32(acc);
            for (size_t j = 0; j < (dims[1]%4); ++j) {
                (*outPtr) += *(tPtr);
                tPtr += 1;
            }
            outPtr += 1;
        }

        Matrix ret = Matrix(new_dims, 1, data_out, false);
        return ret;

    } else {
        throw std::invalid_argument("Invalid dimensions, must be 2D");
    }
}

//General - can improve in future to sum over a given axes, would require continous memory for given axes in order to use SIMD
float Matrix::sum() const {
    float* tPtr = data;
    float out = 0;

    for (size_t i = 0; i + 3 < data_len; i += 4) {
        float32x4_t tf = vld1q_f32(tPtr);
        out += vaddvq_f32(tf);
        tPtr += 4;
    }

    for (size_t i = 0; i < data_len % 4; ++i) {
        out += tPtr[i];
    }

    return out;
}

float Matrix::get(const std::initializer_list<int>& pos) const {
    return data[convert_idx(pos)];
}

float Matrix::get_index(int i) const {
    if (i < 0 || i > data_len-1){
        throw std::invalid_argument("Invalid index!");
    }
    return data[i];
}

void Matrix::set(const std::initializer_list<int>& pos, float val) {
    data[convert_idx(pos)] = val;
}

void Matrix::set_index(int i, float val) {
    if (i < 0 || i > data_len-1) {
        throw std::invalid_argument("Invalid index!");
    }
    data[i] = val;
}

int Matrix::get_dims_index(int i) const {
    if (i < 0 || i > dim_len-1) {
        throw std::invalid_argument("Invalid index!");
    }
    return dims[i];
}

int Matrix::get_dim_len() const {
    return dim_len;
}

int* Matrix::get_dists_clone() const {
    int* dists_clone = (int*) malloc(dim_len * sizeof(int));
    if (dists_clone == nullptr) {
        throw std::invalid_argument("Memory allocation error");
    }
    for (size_t i = 0; i < dim_len; ++i) {
        dists_clone[i] = dists[i];
    }
    return dists_clone;
}

int* Matrix::get_dims_clone() const {
    int* dims_clone = (int*) malloc(dim_len * sizeof(int));
    if (dims_clone == nullptr) {
        throw std::invalid_argument("Memory allocation error");
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

int* Matrix::get_dims() const {
    return dims;
}

int* Matrix::get_dists() const {
    return dists;
}

float* Matrix::get_data() const {
    return data;
}

void Matrix::print_data(int max) const {
    print_array(data, data_len, max);
}

void Matrix::print_dims(int max) const {
    print_array(dims, dim_len, max);
}

void Matrix::print_dists(int max) const {
    print_array(dists, dim_len, max);
}

//Get/Set CUDA Usage
void Matrix::set_CUDA(bool c) {
    cuda = c;
}

bool Matrix::get_CUDA() {
    return cuda;
}

void Matrix::set_tile(int t) {
    tile = t;
}

int Matrix::get_tile() {
    return tile;
}