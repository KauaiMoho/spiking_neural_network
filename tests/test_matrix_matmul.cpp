#include <gtest/gtest.h>
#include "Matrix.h"
#include "test_utils.h"

TEST(MatrixMatmul, DotProduct) {

    int dims[] = {4};

    Matrix A(dims, 1, 1.0f);
    Matrix B(dims, 1, 2.0f);
    Matrix C = A.matmul(B);

    ASSERT_EQ(C.get_dim_len(), 1);
    EXPECT_FLOAT_EQ(C.get_index(0), 8.0f);
}

TEST(MatrixMatmul, VectorTimesMatrix) {

    int dimsA[] = {3};
    int dimsB[] = {3, 2};

    Matrix A(dimsA, 1, 1.0f);
    Matrix B(dimsB, 2, 0.0f);
    fill_sequential(B);

    Matrix C = A.matmul(B);

    EXPECT_EQ(C.get_dim_len(), 1);
    EXPECT_FLOAT_EQ(C.get_index(0), 6.0f);
    EXPECT_FLOAT_EQ(C.get_index(1), 9.0f);
}

TEST(MatrixMatmul, MatrixTimesVector) {

    int dimsA[] = {2, 3};
    int dimsB[] = {3};

    Matrix A(dimsA, 2, 0.0f);
    fill_sequential(A);
    Matrix B(dimsB, 1, 1.0f);

    Matrix C = A.matmul(B);

    EXPECT_EQ(C.get_dim_len(), 1);
    EXPECT_FLOAT_EQ(C.get_index(0), 3.0f);
    EXPECT_FLOAT_EQ(C.get_index(1), 12.0f);
}

TEST(MatrixMatmul, MatrixTimesMatrix) {

    int dimsA[] = {2, 3};
    int dimsB[] = {3, 2};

    Matrix A(dimsA, 2, 0.0f);
    fill_sequential(A);
    Matrix B(dimsB, 2, 1.0f);

    Matrix C = A.matmul(B);

    EXPECT_EQ(C.get_dim_len(), 2);
    EXPECT_FLOAT_EQ(C.get({0, 0}), 3.0f);
    EXPECT_FLOAT_EQ(C.get({0, 1}), 3.0f);
    EXPECT_FLOAT_EQ(C.get({1, 0}), 12.0f);
    EXPECT_FLOAT_EQ(C.get({1, 1}), 12.0f);

    Matrix expected = naive_matmul(A, B);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(C.get({i, j}), expected.get({i, j}));
        }
    }
}

TEST(MatrixMatmul, BatchedMatrixMultiplication) {

    int dimsA[] = {2, 2, 3};
    int dimsB[] = {2, 3, 2};

    Matrix A(dimsA, 3, 0.0f);
    Matrix B(dimsB, 3, 0.0f);

    fill_sequential(A);
    fill_sequential(B);

    Matrix C = A.matmul(B);
    EXPECT_EQ(C.get_dim_len(), 3);
    EXPECT_EQ(C.get_dims_index(0), 2);
    EXPECT_EQ(C.get_dims_index(1), 2);
    EXPECT_EQ(C.get_dims_index(2), 2);

    int dims0[2] = {2, 3};
    int dims1[2] = {3, 2};

    Matrix a0(dims0, 2, 0.0f);
    Matrix b0(dims1, 2, 0.0f);
    fill_sequential(a0);
    fill_sequential(b0);

    Matrix expected = naive_matmul(a0, b0);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(C.get({0, i, j}), expected.get({i, j}));
        }
    }
}
        

TEST(MatrixMatmul, TwoBatch2x3x2x3) {

    int dimsA[] = {2, 2, 3};
    int dimsB[] = {2, 3, 2};

    Matrix A(dimsA, 3, 0.0f);
    Matrix B(dimsB, 3, 0.0f);

    fill_sequential(A);
    fill_sequential(B);

    Matrix C = A.matmul(B);
    EXPECT_EQ(C.get_dim_len(), 3);
    EXPECT_EQ(C.get_dims_index(0), 2);
    EXPECT_EQ(C.get_dims_index(1), 2);
    EXPECT_EQ(C.get_dims_index(2), 2);


    for (int batch = 0; batch < 2; ++batch) {

        int dims0[2] = {2, 3};
        int dims1[2] = {3, 2};

        Matrix a_batch(dims0, 2, 0.0f);
        Matrix b_batch(dims1, 2, 0.0f);

        for (int i = 0; i < 6; ++i) {
            a_batch.set_index(i, A.get_index(batch*6 + i));
        }
        for (int i = 0; i < 6; ++i) {
            b_batch.set_index(i, B.get_index(batch*6 + i));
        }

        Matrix expected = naive_matmul(a_batch, b_batch);
        
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                EXPECT_FLOAT_EQ(C.get({batch, i, j}), expected.get({i, j}));
            }
        }
    }
}

TEST(MatrixMatmul, BroadcastBatch) {

    int dimsA[] = {1, 2, 3};
    int dimsB[] = {2, 3, 2};

    Matrix A(dimsA, 3, 0.0f);
    Matrix B(dimsB, 3, 0.0f);

    fill_sequential(A);
    fill_sequential(B);

    Matrix C = A.matmul(B);
    EXPECT_EQ(C.get_dim_len(), 3);
    EXPECT_EQ(C.get_dims_index(0), 2);
    EXPECT_EQ(C.get_dims_index(1), 2);
    EXPECT_EQ(C.get_dims_index(2), 2);


    for (int batch = 0; batch < 2; ++batch) {

        int dims0[2] = {2, 3};
        int dims1[2] = {3, 2};

        Matrix a_batch(dims0, 2, 0.0f);
        Matrix b_batch(dims1, 2, 0.0f);

        fill_sequential(a_batch);
        for (int i = 0; i < 6; ++i) {
            b_batch.set_index(i, B.get_index(batch*6 + i));
        }

        Matrix expected = naive_matmul(a_batch, b_batch);

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                EXPECT_FLOAT_EQ(C.get({batch, i, j}), expected.get({i, j}));
            }
        }
    }
}

TEST(MatrixMatmul, LargeBMM) {

    int dimsA[] = {4, 5, 8};
    int dimsB[] = {4, 8, 3};

    Matrix A(dimsA, 3, 0.0f);
    Matrix B(dimsB, 3, 0.0f);

    fill_sequential(A);
    fill_sequential(B);

    Matrix C = A.matmul(B);
    EXPECT_EQ(C.get_dim_len(), 3);
    EXPECT_EQ(C.get_dims_index(0), 4);
    EXPECT_EQ(C.get_dims_index(1), 5);
    EXPECT_EQ(C.get_dims_index(2), 3);
}

TEST(MatrixMatmul, ScalarBMM) {

    int dimsA[] = {1, 1, 1};
    int dimsB[] = {1, 1, 1};

    Matrix A(dimsA, 3, 0.0f);
    Matrix B(dimsB, 3, 0.0f);
    A.set_index(0, 2.5f);
    B.set_index(0, 4.0f);

    Matrix C = A.matmul(B);

    EXPECT_EQ(C.get_dim_len(), 3);
    EXPECT_EQ(C.get_dims_index(0), 1);
    EXPECT_EQ(C.get_dims_index(1), 1);
    EXPECT_EQ(C.get_dims_index(2), 1);
    EXPECT_FLOAT_EQ(C.get_index(0), 10.0f);
}

TEST(MatrixMatmul, NonSquareSingleBatch) {

    int dimsA[] = {1, 7, 13};
    int dimsB[] = {1, 13, 4};

    Matrix A(dimsA, 3, 0.0f);
    Matrix B(dimsB, 3, 0.0f);

    fill_sequential(A);
    fill_sequential(B);

    Matrix C = A.matmul(B);

    EXPECT_EQ(C.get_dim_len(), 3);
    EXPECT_EQ(C.get_dims_index(0), 1);
    EXPECT_EQ(C.get_dims_index(1), 7);
    EXPECT_EQ(C.get_dims_index(2), 4);
}