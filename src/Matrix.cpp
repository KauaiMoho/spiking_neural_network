#include "Matrix.h"
using namespace std;

Matrix::Matrix(vector<int> d) : dims(d) {
    //Define the dists var here

    if (d.empty()) throw std::invalid_argument("Matrix dimensions cannot be empty!");

    dists.resize(d.size());
    int pos = 1;
    for (int i = d.size()-1; i > 0 ; i--) {
        dists[i] = pos;
        pos *= dims[i];
    }
    data.resize(pos*d[d.size()-1]);
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

Matrix Matrix::matmul(Matrix a) {

}

Matrix Matrix::scmul(Matrix a) {

}

Matrix Matrix::add(Matrix a) {

}

Matrix Matrix::subtract(Matrix a) {

}

Matrix Matrix::transpose() {

}

float Matrix::get(vector<int> pos) {
    return data[get_idx(pos)];
}

void Matrix::set(vector<int> pos, float val) {
    data[get_idx(pos)] = val;
}
