#include <gtest/gtest.h>
#include "Tensor.h"
#include "test_utils.h"

TEST(TensorMatmul, DotProduct) {

    int dims[] = {4};

    Tensor A(dims, 1, 1.0f);
    Tensor B(dims, 1, 2.0f);
    Tensor C = A.matmul(B);

    ASSERT_EQ(C.get_dim_len(), 1);
    EXPECT_FLOAT_EQ(C.get_index(0), 8.0f);
}

TEST(TensorMatmul, VectorTimesTensor) {

    int dimsA[] = {3};
    int dimsB[] = {3, 2};

    Tensor A(dimsA, 1, 1.0f);
    Tensor B(dimsB, 2, 0.0f);
    fill_sequential(B);

    Tensor C = A.matmul(B);

    EXPECT_EQ(C.get_dim_len(), 1);
    EXPECT_FLOAT_EQ(C.get_index(0), 6.0f);
    EXPECT_FLOAT_EQ(C.get_index(1), 9.0f);
}

TEST(TensorMatmul, TensorTimesVector) {

    int dimsA[] = {2, 3};
    int dimsB[] = {3};

    Tensor A(dimsA, 2, 0.0f);
    fill_sequential(A);
    Tensor B(dimsB, 1, 1.0f);

    Tensor C = A.matmul(B);

    EXPECT_EQ(C.get_dim_len(), 1);
    EXPECT_FLOAT_EQ(C.get_index(0), 3.0f);
    EXPECT_FLOAT_EQ(C.get_index(1), 12.0f);
}

TEST(TensorMatmul, TensorTimesTensor) {

    int dimsA[] = {2, 3};
    int dimsB[] = {3, 2};

    Tensor A(dimsA, 2, 0.0f);
    fill_sequential(A);
    Tensor B(dimsB, 2, 1.0f);

    Tensor C = A.matmul(B);

    EXPECT_EQ(C.get_dim_len(), 2);
    EXPECT_FLOAT_EQ(C.get({0, 0}), 3.0f);
    EXPECT_FLOAT_EQ(C.get({0, 1}), 3.0f);
    EXPECT_FLOAT_EQ(C.get({1, 0}), 12.0f);
    EXPECT_FLOAT_EQ(C.get({1, 1}), 12.0f);

    Tensor expected = naive_matmul(A, B);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(C.get({i, j}), expected.get({i, j}));
        }
    }
}

TEST(TensorMatmul, BatchedTensorMultiplication) {

    int dimsA[] = {2, 2, 3};
    int dimsB[] = {2, 3, 2};

    Tensor A(dimsA, 3, 0.0f);
    Tensor B(dimsB, 3, 0.0f);

    fill_sequential(A);
    fill_sequential(B);

    Tensor C = A.matmul(B);
    EXPECT_EQ(C.get_dim_len(), 3);
    EXPECT_EQ(C.get_dims_index(0), 2);
    EXPECT_EQ(C.get_dims_index(1), 2);
    EXPECT_EQ(C.get_dims_index(2), 2);

    int dims0[2] = {2, 3};
    int dims1[2] = {3, 2};

    Tensor a0(dims0, 2, 0.0f);
    Tensor b0(dims1, 2, 0.0f);
    fill_sequential(a0);
    fill_sequential(b0);

    Tensor expected = naive_matmul(a0, b0);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(C.get({0, i, j}), expected.get({i, j}));
        }
    }
}
        

TEST(TensorMatmul, TwoBatch2x3x2x3) {

    int dimsA[] = {2, 2, 3};
    int dimsB[] = {2, 3, 2};

    Tensor A(dimsA, 3, 0.0f);
    Tensor B(dimsB, 3, 0.0f);

    fill_sequential(A);
    fill_sequential(B);

    Tensor C = A.matmul(B);
    EXPECT_EQ(C.get_dim_len(), 3);
    EXPECT_EQ(C.get_dims_index(0), 2);
    EXPECT_EQ(C.get_dims_index(1), 2);
    EXPECT_EQ(C.get_dims_index(2), 2);


    for (int batch = 0; batch < 2; ++batch) {

        int dims0[2] = {2, 3};
        int dims1[2] = {3, 2};

        Tensor a_batch(dims0, 2, 0.0f);
        Tensor b_batch(dims1, 2, 0.0f);

        for (int i = 0; i < 6; ++i) {
            a_batch.set_index(i, A.get_index(batch*6 + i));
        }
        for (int i = 0; i < 6; ++i) {
            b_batch.set_index(i, B.get_index(batch*6 + i));
        }

        Tensor expected = naive_matmul(a_batch, b_batch);
        
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                EXPECT_FLOAT_EQ(C.get({batch, i, j}), expected.get({i, j}));
            }
        }
    }
}

TEST(TensorMatmul, BroadcastBatch) {

    int dimsA[] = {1, 2, 3};
    int dimsB[] = {2, 3, 2};

    Tensor A(dimsA, 3, 0.0f);
    Tensor B(dimsB, 3, 0.0f);

    fill_sequential(A);
    fill_sequential(B);

    Tensor C = A.matmul(B);
    EXPECT_EQ(C.get_dim_len(), 3);
    EXPECT_EQ(C.get_dims_index(0), 2);
    EXPECT_EQ(C.get_dims_index(1), 2);
    EXPECT_EQ(C.get_dims_index(2), 2);


    for (int batch = 0; batch < 2; ++batch) {

        int dims0[2] = {2, 3};
        int dims1[2] = {3, 2};

        Tensor a_batch(dims0, 2, 0.0f);
        Tensor b_batch(dims1, 2, 0.0f);

        fill_sequential(a_batch);
        for (int i = 0; i < 6; ++i) {
            b_batch.set_index(i, B.get_index(batch*6 + i));
        }

        Tensor expected = naive_matmul(a_batch, b_batch);

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                EXPECT_FLOAT_EQ(C.get({batch, i, j}), expected.get({i, j}));
            }
        }
    }
}

TEST(TensorMatmul, LargeBMM) {

    int dimsA[] = {4, 5, 8};
    int dimsB[] = {4, 8, 3};

    Tensor A(dimsA, 3, 0.0f);
    Tensor B(dimsB, 3, 0.0f);

    fill_sequential(A);
    fill_sequential(B);

    Tensor C = A.matmul(B);
    EXPECT_EQ(C.get_dim_len(), 3);
    EXPECT_EQ(C.get_dims_index(0), 4);
    EXPECT_EQ(C.get_dims_index(1), 5);
    EXPECT_EQ(C.get_dims_index(2), 3);
}

TEST(TensorMatmul, ScalarBMM) {

    int dimsA[] = {1, 1, 1};
    int dimsB[] = {1, 1, 1};

    Tensor A(dimsA, 3, 0.0f);
    Tensor B(dimsB, 3, 0.0f);
    A.set_index(0, 2.5f);
    B.set_index(0, 4.0f);

    Tensor C = A.matmul(B);

    EXPECT_EQ(C.get_dim_len(), 3);
    EXPECT_EQ(C.get_dims_index(0), 1);
    EXPECT_EQ(C.get_dims_index(1), 1);
    EXPECT_EQ(C.get_dims_index(2), 1);
    EXPECT_FLOAT_EQ(C.get_index(0), 10.0f);
}

TEST(TensorMatmul, NonSquareSingleBatch) {

    int dimsA[] = {1, 7, 13};
    int dimsB[] = {1, 13, 4};

    Tensor A(dimsA, 3, 0.0f);
    Tensor B(dimsB, 3, 0.0f);

    fill_sequential(A);
    fill_sequential(B);

    Tensor C = A.matmul(B);

    EXPECT_EQ(C.get_dim_len(), 3);
    EXPECT_EQ(C.get_dims_index(0), 1);
    EXPECT_EQ(C.get_dims_index(1), 7);
    EXPECT_EQ(C.get_dims_index(2), 4);
}

TEST(TensorMatmul, Square32) {
    int dimsA[] = {32, 32};
    int dimsB[] = {32, 32};

    Tensor A(dimsA, 2, 0.0f);
    Tensor B(dimsB, 2, 0.0f);

    fill_sequential(A);
    fill_sequential(B);

    Tensor C_ref = naive_matmul(A, B);
    Tensor C_amx = A.matmul(B);

    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 32; j++)
            EXPECT_NEAR(C_amx.get({i, j}), C_ref.get({i, j}), 1e-3f)
                << "Mismatch at (" << i << ", " << j << ")";
}

TEST(TensorMatmul, Square48) {
    int dimsA[] = {48, 48};
    int dimsB[] = {48, 48};

    Tensor A(dimsA, 2, 0.0f);
    Tensor B(dimsB, 2, 0.0f);

    fill_sequential(A);
    fill_sequential(B);

    Tensor C_ref = naive_matmul(A, B);
    Tensor C_amx = A.matmul(B);

    for (int i = 0; i < 48; i++)
        for (int j = 0; j < 48; j++)
            EXPECT_NEAR(C_amx.get({i, j}), C_ref.get({i, j}), 1e-3f)
                << "Mismatch at (" << i << ", " << j << ")";
}

TEST(TensorMatmul, Square49) {
    int dimsA[] = {49, 49};
    int dimsB[] = {49, 49};

    Tensor A(dimsA, 2, 0.0f);
    Tensor B(dimsB, 2, 0.0f);

    fill_sequential(A);
    fill_sequential(B);

    Tensor C_ref = naive_matmul(A, B);
    Tensor C_amx = A.matmul(B);

    for (int i = 0; i < 49; i++)
        for (int j = 0; j < 49; j++)
            EXPECT_NEAR(C_amx.get({i, j}), C_ref.get({i, j}), 1e-3f)
                << "Mismatch at (" << i << ", " << j << ")";
}