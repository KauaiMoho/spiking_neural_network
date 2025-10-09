#include "Matrix.h"
using namespace std;
using std::initializer_list;
#include <stdexcept>
#include <cstdlib>
#include <iostream>

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

    data = (float*) malloc(data_len * sizeof(float));
    if (data == nullptr) {
        throw invalid_argument("Memory allocation error");
    }

    for (int i = 0; i < data_len; i++) {
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

    data = (float*) malloc(data_len * sizeof(float));
    if (data == nullptr) {
        throw invalid_argument("Memory allocation error");
    }

    for (int i = 0; i < data_len; i++) {
        data[i] = val;
    }
}

int Matrix::convert_idx(initializer_list<int> pos) {
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



int* Matrix::get_broadcasted_strides(int* dims_new, int dim_len_new) {
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
    for (int i = 0; i < dim_len_new; i++) {
        dim_size_old *= dims[i];
    }
    int dim_size_new = 1;
    for (int i = 0; i < dim_len_new; i++) {
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
    if (!(dim_len_new >= dim_len)) {
        int* dists_new = get_broadcasted_strides(dims_new, dim_len_new);
        free(dists);
        free(dims);
        dists = dists_new;
        dims = (int*) malloc(dim_len_new * sizeof(int));
        if (dims == nullptr) {
            throw invalid_argument("Memory allocation error");
        }
        for (int i = 0; i < dim_len_new; i++) {
            dims[i] = dims_new[i];
        }
        dim_len = dim_len_new;
    } else {
        throw invalid_argument("Invalid dimension size for broadcasting!");
    }
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

    //Assumption of near square matrices
    int tile = static_cast<int>(sqrt(usable_cache_floats / 3));
    tile = tile & ~(cache_line_floats - 1);
    if (tile == 0) {
        tile = cache_line_floats;
    }

    for (int ic = 0; ic < n; ic += tile){
        for (int jc = 0; jc < m; jc += tile){
            for (int lc = 0; lc < k; lc += tile){
                for (int i = ic; i < min(ic+tile, n); i++){
                    for (int j = jc; j < min(jc+tile, m); j++) {
                        float first = A[i*m + j];
                        for (int l = lc; l < min(lc+tile, k); l++){
                            C[i*k + l] += first * B[j*k + l];
                        }
                    }
                }
            }
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
            for (int i = 0; i < data_len; i++) {
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
            for (int i = 0; i < new_dims[0]; i++) {
                float sum = 0;
                for (int j = 0; j < dims[1]; j++) {
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
            for (int i = 0; i < new_dims[0]; i++) {
                float sum = 0;
                for (int j = 0; j < dims[0]; j++) {
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
                            broadcast_dims[i] = dims[i];
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
            bmm_shape[1] = dims[dim_len-2];
            bmm_shape[2] = other.get_dims_index(other_dim_len - 1);
            for (int i = 0; i < broadcast_dim_len - 2; i++) {
                bmm_shape[0] *= broadcast_dims[i];
            }

            //Save old dimensions, do not free
            int dim_len_this = dim_len;
            int dim_len_other = other.get_dim_len();
            int* dims_clone_this = get_dims_clone();
            int* dims_clone_other = other.get_dims_clone();
            int* dists_clone_this = get_dists_clone();
            int* dists_clone_other = other.get_dists_clone();
            
            //Broadcast and reshape, might want to make it so this can be done outside - seperate function

            //NEED TO MAKE SURE LAST TWO DIMS UNTOUCHED IN BROADCAST
            broadcast(broadcast_dims, broadcast_dim_len);
            other.broadcast(broadcast_dims, broadcast_dim_len);
            reshape(bmm_shape, 3);
            other.reshape(bmm_shape, 3);

            //NOW MATMUL


            dim_len = dim_len_this;
            dims = dims_clone_this;
            dists = dists_clone_this;
            other.set_dim_len(dim_len_other);
            other.set_dims(dims_clone_other);
            other.set_dists(dists_clone_other);

        } else {
            throw invalid_argument("Invalid batched matrix-matrix product dimensions!");
        }
    }
}   

Matrix Matrix::clone() {
    return Matrix(dims, dim_len, data);
}

void Matrix::scmul(float s) {
    for (int i = 0; i < data_len; i++) {
        data[i] = data[i] * s;
    }
}

void Matrix::add(Matrix a) {
    for (int i = 0; i < dim_len; i++){
        if (dims[i] != a.get_dims_index(i)) {
            throw invalid_argument("Invalid matrix dimensions!");
        }
    }
    for (int i = 0; i < data_len; i++) {
        data[i] = data[i] + a.get_index(i);
    }
}

void Matrix::subtract(Matrix a) {
    for (int i = 0; i < dim_len; i++){
        if (dims[i] != a.get_dims_index(i)) {
            throw invalid_argument("Invalid matrix dimensions!");
        }
    }
    for (int i = 0; i < data_len; i++) {
        data[i] = data[i] - a.get_index(i);
    }
}

void Matrix::transpose(int* axes) {
    int* dists_c = (int*) malloc(dim_len * sizeof(int));
    if (dists_c == nullptr) {
        throw invalid_argument("Memory allocation error");
    }
    for (int i = 0; i < dim_len; i++) {
        dists_c[i] = dists[axes[i]];
    }
    for (int i = 0; i < dim_len; i++) {
        dists[i] = dists_c[i];
    }
    free(dists_c);
}

float Matrix::get(initializer_list<int> pos) {
    return data[convert_idx(pos)];
}

float Matrix::get_index(int i) {
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

int Matrix::get_dims_index(int i) {
    if (i < 0 || i > dim_len-1) {
        throw invalid_argument("Invalid index!");
    }
    return dims[i];
}

int Matrix::get_dim_len() {
    return dim_len;
}

int* Matrix::get_dists_clone() {
    int* dists_clone = (int*) malloc(dim_len * sizeof(int));
    if (dists_clone == nullptr) {
        throw invalid_argument("Memory allocation error");
    }
    for (int i = 0; i < dim_len; i++) {
        dists_clone[i] = dists[i];
    }
    return dists_clone;
}

int* Matrix::get_dims_clone() {
    int* dims_clone = (int*) malloc(dim_len * sizeof(int));
    if (dims_clone == nullptr) {
        throw invalid_argument("Memory allocation error");
    }
    for (int i = 0; i < dim_len; i++) {
        dims_clone[i] = dims[i];
    }
    return dims_clone;
}

void Matrix::set_dim_len(int dim_len_n) {
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

float* Matrix::get_data() {
    return data;
}

void Matrix::set_CUDA(bool c) {
    cuda = c;
}

bool Matrix::get_CUDA() {
    return cuda;
}