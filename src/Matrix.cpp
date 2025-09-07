#include "Matrix.h"
using namespace std;
#include <stdexcept>
#include <cstdlib>

Matrix::Matrix(int* dims_n, int dim_len, float* data_n, int data_len) : dim_len(dim_len), data_len(data_len) {
    //Define the dists var here
    if (dim_len == 0) throw invalid_argument("Matrix dimensions cannot be empty!");
    dists = (int*) malloc(dim_len * sizeof(int));

    if (dists == nullptr) {
        throw invalid_argument("Memory allocation error");
    }

    dims = (int*) malloc(dim_len * sizeof(int));

    if (dims == nullptr) {
        throw invalid_argument("Memory allocation error");
    }

    data = (float*) malloc(data_len * sizeof(float));

    if (data == nullptr) {
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

    data = (float*) malloc(data_len * sizeof(float));

    if (data == nullptr) {
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
    for (int i = 0; i < data_len; i++) {
        data[i] = val;
    }
}

int Matrix::get_idx(int* pos) {
    int idx = 0;
    for (int i = 0; i < dim_len; i++) {
        idx += pos[i]*dists[i];
    }
    if (idx > data_len || idx < 0) throw invalid_argument("Invalid position dimensions!");
    return idx;
}

Matrix Matrix::matmul(int* this_axes, int len_this_a, Matrix other, int*other_axes, int len_other_a) {
    if (len_this_a != len_other_a){
        throw invalid_argument("Invalid axes dimensions!");
    }
    for (int i = 0; i < len_this_a; i++) {
        if (dims[this_axes[i]] != other.get_dims_index(other_axes[i])) {
            throw invalid_argument("Invalid axes values!");
        }
    }
    //implement matrix multiplaction here
}   

void Matrix::scmul(float s) {
    for (int i = 0; i < data_len; i++) {
        data[i] = data[i] * s;
    }
}

void Matrix::add(Matrix a) {
    if (a.get_full_dims() == dims) {
        for (int i = 0; i < dim_len; i++) {
            data[i] = data[i] + a.get_index(i);
        }
    } else {
        throw invalid_argument("Invalid matrix dimensions!");
    }
}

void Matrix::subtract(Matrix a) {
    if (a.get_full_dims() == dims) {
        for (int i = 0; i < dim_len; i++) {
            data[i] = data[i] - a.get_index(i);
        }
    } else {
        throw invalid_argument("Invalid matrix dimensions!");
    }
}

void Matrix::transpose() {

}

float Matrix::get(int* pos) {
    return data[get_idx(pos)];
}

void Matrix::set(int* pos, float val) {
    data[get_idx(pos)] = val;
}

int* Matrix::get_full_dims() {
    return dims;
}

int Matrix::get_dims_index(int i) {
    if (i < 0 || i > dim_len-1){
        throw invalid_argument("Invalid index!");
    }
    return dims[i];
}

float Matrix::get_index(int i) {
    if (i < 0 || i > dim_len-1){
        throw invalid_argument("Invalid index!");
    }
    return data[i];
}
