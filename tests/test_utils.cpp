#include "Matrix.h"
#include "test_utils.h"

void fill_sequential(Matrix &m) {
    int dim_len = m.get_dim_len();
    int total = 1;
    for (int i = 0; i < dim_len; ++i)
        total *= m.get_dims_index(i);
    for (int i = 0; i < total; ++i) {
        m.set_index(i, static_cast<float>(i));
    }
}

float square(float s) {
    return s * s;
}

Matrix naive_matmul(const Matrix &A, const Matrix &B) {
    
    int n = A.get_dims_index(0);
    int m = A.get_dims_index(1);
    int k = B.get_dims_index(1);

    int dims[] = {n, k};
    Matrix C(dims, 2, 0.0f);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            float sum = 0;
            for (int l = 0; l < m; ++l)
                sum += A.get({i, l}) * B.get({l, j});
            C.set({i, j}, sum);
        }
    }

    return C;
}