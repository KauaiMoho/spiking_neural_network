#include "Matrix.h"
using namespace std;
#include <stdexcept>
#include <cstdlib>
#include <iostream>

Matrix::Matrix(int* dims_n, int dim_len, float* data_n) : dim_len(dim_len) {
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

    data = (float*) malloc(data_len * sizeof(float));

    if (data == nullptr) {
        throw invalid_argument("Memory allocation error");
    }

    dims[0] = dims_n[0];
    dists[0] = pos;
    data_len = pos*dims[0];


    for (int i = 0; i < data_len; i++) {
        data[i] = val;
    }
}

int Matrix::convert_idx(int* pos) {
    int idx = 0;
    for (int i = 0; i < dim_len; i++) {
        idx += pos[i]*dists[i];
    }
    if (idx > data_len - 1 || idx < 0) throw invalid_argument("Invalid position dimensions!");
    return idx;
}

Matrix Matrix::matmul(int* this_axes, Matrix other, int*other_axes, int len_a) {
    for (int i = 0; i < len_a; i++) {
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

float Matrix::get(int* pos) {
    return data[convert_idx(pos)];
}

float Matrix::get_index(int i) {
    if (i < 0 || i > data_len-1){
        throw invalid_argument("Invalid index!");
    }
    return data[i];
}

void Matrix::set(int* pos, float val) {
    data[convert_idx(pos)] = val;
}

void Matrix::set_index(int i, float val) {
    if (i < 0 || i > data_len-1){
        throw invalid_argument("Invalid index!");
    }
    data[i] = val;
}

int Matrix::get_dims_index(int i) {
    if (i < 0 || i > dim_len-1){
        throw invalid_argument("Invalid index!");
    }
    return dims[i];
}
