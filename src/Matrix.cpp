#include "Matrix.h"
using namespace std;

Matrix::Matrix(vector<int> dims, vector<float> data) : dims(dims), data(data) {
    //Define the dists var here

    if (dims.empty()) throw std::invalid_argument("Matrix dimensions cannot be empty!");

    dists.resize(dims.size());
    int pos = 1;
    for (int i = dims.size()-1; i > 0 ; i--) {
        dists[i] = pos;
        pos *= dims[i];
    }
}

Matrix::Matrix(vector<int> dims) : dims(dims) {
    //Define the dists var here

    if (dims.empty()) throw std::invalid_argument("Matrix dimensions cannot be empty!");

    dists.resize(dims.size());
    int pos = 1;
    for (int i = dims.size()-1; i > 0 ; i--) {
        dists[i] = pos;
        pos *= dims[i];
    }
    data.resize(pos*dims[dims.size()-1]);
}

int Matrix::get_idx(vector<int> pos) {
    int idx = 0;

    if (pos.size() != dists.size()) throw std::invalid_argument("Invalid position dimensions!");

    for (int i = 0; i < dists.size(); i++) {
        idx += pos[i]*dists[i];
    }

    if (idx > data.size() || idx < 0) throw std::invalid_argument("Invalid position dimensions!");
    
    return idx;
}

Matrix Matrix::matmul(vector<int> this_axes, Matrix other, vector<int> other_axes) {
    if (this_axes.size() != other_axes.size()){
        throw std::invalid_argument("Invalid axes dimensions!");
    }
    for (int i = 0; i < this_axes.size(); i++) {
        if (dims[this_axes[i]] != other.get_dims_index(other_axes[i])) {
            throw std::invalid_argument("Invalid axes values!");
        }
    }
    //implement matrix multiplaction here
}   

void Matrix::scmul(float s) {
    for (int i = 0; i < data.size(); i++) {
        data[i] = data[i] * s;
    }
}

void Matrix::add(Matrix a) {
    if (a.get_full_dims() == dims) {
        for (int i = 0; i < data.size(); i++) {
            data[i] = data[i] + a.get_index(i);
        }
    } else {
        throw std::invalid_argument("Invalid matrix dimensions!");
    }
}

void Matrix::subtract(Matrix a) {
    if (a.get_full_dims() == dims) {
        for (int i = 0; i < data.size(); i++) {
            data[i] = data[i] - a.get_index(i);
        }
    } else {
        throw std::invalid_argument("Invalid matrix dimensions!");
    }
}

void Matrix::transpose() {

}

float Matrix::get(vector<int> pos) {
    return data[get_idx(pos)];
}

void Matrix::set(vector<int> pos, float val) {
    data[get_idx(pos)] = val;
}

vector<int> Matrix::get_full_dims() {
    return dims;
}

int Matrix::get_dims_index(int i) {
    if (i < 0 || i > dims.size()-1){
        throw std::invalid_argument("Invalid index!");
    }
    return dims[i];
}

float Matrix::get_index(int i) {
    if (i < 0 || i > data.size()-1){
        throw std::invalid_argument("Invalid index!");
    }
    return data[i];
}
