#include <gtest/gtest.h>
#include "Matrix.h"
#include "test_utils.h"

TEST(MatrixOps, Clone) {

    int dims[] = {3, 3};

    Matrix A(dims,2,0.0f);
    fill_sequential(A);
    Matrix B = A.clone();
    
    EXPECT_EQ(B.get_dim_len(), A.get_dim_len());

    for (int i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(B.get_index(i), A.get_index(i));
    }

    B.set_index(0, 999.0f);
    
    EXPECT_FLOAT_EQ(A.get_index(0), 0.0f);
}

TEST(MatrixOps, ScMul) {

    int dims[] = {2, 3};

    Matrix A(dims,2,1.0f);
    Matrix B = A.scmul(3.0f);

    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(B.get_index(i), 3.0f);
    }
}

TEST(MatrixOps, ScMulInPlace) {

    int dims[] = {2, 3};

    Matrix A(dims,2,2.0f);
    A.scmul_inplace(4.0f);

    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(A.get_index(i), 8.0f);
    }
}


TEST(MatrixOps, Apply) {

    int dims[] = {2, 2};

    Matrix A(dims,2,2.0f);
    Matrix B = A.apply(square);

    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(B.get_index(i), 4.0f);
    }

    A.apply_inplace(square);

    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(A.get_index(i), 4.0f);
    }
}

TEST(MatrixOps, AddSmall_ColVector) {

    int dims[] = {3, 3};
    int vdims[] = {3};

    Matrix A(dims, 2, 1.0f);
    Matrix B(vdims, 1, 2.0f);

    Matrix C = A.add(B);
    for (int i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(C.get_index(i), 3.0f);
    }

    A.add_inplace(B);
    for (int i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(A.get_index(i), 3.0f);
    }
}


TEST(MatrixOps, AddSmall_RowVector) {

    int dims[] = {2, 5};
    int vdims[] = {5};

    Matrix A(dims, 2, 1.0f);
    Matrix B(vdims, 1, 1.5f);
    Matrix C = A.add(B);

    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(C.get_index(i), 2.5f);
    }

    A.add_inplace(B);
    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(A.get_index(i), 2.5f);
    }
}

TEST(MatrixOps, AddSmall_MatrixMatrix) {

    int dims[] = {2, 3};

    Matrix A(dims, 2, 1.0f);
    Matrix B(dims, 2, 4.0f);
    Matrix C = A.add(B);

    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(C.get_index(i), 5.0f);
    }

    A.add_inplace(B);

    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(A.get_index(i), 5.0f);
    }

}

TEST(MatrixOps, AddLarge_ColVector) {

    int dims[] = {128, 256};
    int vdims[] = {128};

    Matrix A(dims, 2, 1.0f);
    Matrix B(vdims, 1, 3.0f);
    Matrix C = A.add(B);

    for (int i = 0; i < 128 * 256; ++i) {
        EXPECT_FLOAT_EQ(C.get_index(i), 4.0f);
    }

    A.add_inplace(B);

    for (int i = 0; i < 128 * 256; ++i) {
        EXPECT_FLOAT_EQ(A.get_index(i), 4.0f);
    }
}

TEST(MatrixOps, AddLarge_RowVector) {

    int dims[] = {64, 128};
    int vdims[] = {128};

    Matrix A(dims, 2, 2.0f);
    Matrix B(vdims, 1, 0.5f);
    Matrix C = A.add(B);

    for (int i = 0; i < 64 * 128; ++i) {
        EXPECT_FLOAT_EQ(C.get_index(i), 2.5f);
    }

    A.add_inplace(B);

    for (int i = 0; i < 64 * 128; ++i) {
        EXPECT_FLOAT_EQ(A.get_index(i), 2.5f);
    }
}

TEST(MatrixOps, AddLarge_MatrixMatrix) {

    int dims[] = {256, 256};

    Matrix A(dims, 2, 1.25f);
    Matrix B(dims, 2, 2.75f);

    Matrix C = A.add(B);

    for (int i = 0; i < 256 * 256; ++i) {
        EXPECT_FLOAT_EQ(C.get_index(i), 4.0f);
    }

    A.add_inplace(B);

    for (int i = 0; i < 256 * 256; ++i) {
        EXPECT_FLOAT_EQ(A.get_index(i), 4.0f);
    }
}

TEST(MatrixOps, AddLarge_NonAlignedDims_RowVector) {

    int dims[]  = {127, 251};
    int vdims[] = {251};

    Matrix A(dims, 2, 1.0f);
    Matrix B(vdims, 1, 2.25f);

    Matrix C = A.add(B);
    for (int i = 0; i < 127 * 251; ++i) {
        EXPECT_FLOAT_EQ(C.get_index(i), 3.25f);
    }

    A.add_inplace(B);

    for (int i = 0; i < 127 * 251; ++i) {
        EXPECT_FLOAT_EQ(A.get_index(i), 3.25f);
    }
}


TEST(MatrixOps, EMulSmall_ColVector) {

    int dims[] = {3, 3};
    int vdims[] = {3};

    Matrix A(dims, 2, 1.0f);
    Matrix B(vdims, 1, 2.0f);

    Matrix C = A.emul(B);
    for (int i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(C.get_index(i), 2.0f);
    }

    A.emul_inplace(B);
    for (int i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(A.get_index(i), 2.0f);
    }
}

TEST(MatrixOps, EMulSmall_RowVector) {

    int dims[] = {2, 5};
    int vdims[] = {5};

    Matrix A(dims, 2, 1.0f);
    Matrix B(vdims, 1, 1.5f);
    Matrix C = A.emul(B);

    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(C.get_index(i), 1.5f);
    }

    A.emul_inplace(B);
    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(A.get_index(i), 1.5f);
    }
}

TEST(MatrixOps, EMulSmall_MatrixMatrix) {

    int dims[] = {2, 3};

    Matrix A(dims, 2, 1.0f);
    Matrix B(dims, 2, 4.0f);
    Matrix C = A.emul(B);

    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(C.get_index(i), 4.0f);
    }

    A.emul_inplace(B);

    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(A.get_index(i), 4.0f);
    }

}

TEST(MatrixOps, EMulLarge_ColVector) {

    int dims[] = {128, 256};
    int vdims[] = {128};

    Matrix A(dims, 2, 1.0f);
    Matrix B(vdims, 1, 3.0f);
    Matrix C = A.emul(B);

    for (int i = 0; i < 128 * 256; ++i) {
        EXPECT_FLOAT_EQ(C.get_index(i), 3.0f);
    }

    A.emul_inplace(B);

    for (int i = 0; i < 128 * 256; ++i) {
        EXPECT_FLOAT_EQ(A.get_index(i), 3.0f);
    }
}

TEST(MatrixOps, EMulLarge_RowVector) {

    int dims[] = {64, 128};
    int vdims[] = {128};

    Matrix A(dims, 2, 2.0f);
    Matrix B(vdims, 1, 0.5f);
    Matrix C = A.emul(B);

    for (int i = 0; i < 64 * 128; ++i) {
        EXPECT_FLOAT_EQ(C.get_index(i), 1.0f);
    }

    A.emul_inplace(B);

    for (int i = 0; i < 64 * 128; ++i) {
        EXPECT_FLOAT_EQ(A.get_index(i), 1.0f);
    }
}

TEST(MatrixOps, EMulLarge_MatrixMatrix) {

    int dims[] = {256, 256};

    Matrix A(dims, 2, 1.25f);
    Matrix B(dims, 2, 2.75f);

    Matrix C = A.emul(B);

    for (int i = 0; i < 256 * 256; ++i) {
        EXPECT_FLOAT_EQ(C.get_index(i), 3.4375f);
    }

    A.emul_inplace(B);

    for (int i = 0; i < 256 * 256; ++i) {
        EXPECT_FLOAT_EQ(A.get_index(i), 3.4375f);
    }
}

TEST(MatrixOps, EMulLarge_NonAlignedDims_RowVector) {

    int dims[]  = {127, 251};
    int vdims[] = {251};

    Matrix A(dims, 2, 1.0f);
    Matrix B(vdims, 1, 2.25f);

    Matrix C = A.emul(B);
    for (int i = 0; i < 127 * 251; ++i) {
        EXPECT_FLOAT_EQ(C.get_index(i), 2.25f);
    }

    A.emul_inplace(B);

    for (int i = 0; i < 127 * 251; ++i) {
        EXPECT_FLOAT_EQ(A.get_index(i), 2.25f);
    }
}


TEST(MatrixOps, Transpose2D) {

    int dims[] = {2, 3};

    Matrix A(dims,2,0.0f);
    fill_sequential(A);
    Matrix B = A.transpose2d();

    EXPECT_EQ(B.get_dims_index(0), 3);
    EXPECT_EQ(B.get_dims_index(1), 2);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(A.get({i,j}), B.get({j,i}));   
        }
    }
}

TEST(MatrixOps, MediumTranspose2D) {

    int dims[] = {20, 20};
    Matrix A(dims, 2, 0.0f);
    fill_sequential(A);

    Matrix At = A.transpose2d();

    EXPECT_EQ(At.get_dims_index(0), dims[1]);
    EXPECT_EQ(At.get_dims_index(1), dims[0]);

    for (int row = 0; row < 5; ++row) {
        for (int col = 0; col < 5; ++col) {
            float original = A.get({row, col});
            float transposed = At.get({col, row});
            EXPECT_FLOAT_EQ(original, transposed);
        }
    }

    EXPECT_FLOAT_EQ(A.get({0,0}), At.get({0,0}));
    EXPECT_FLOAT_EQ(A.get({19,19}), At.get({19,19}));
}

TEST(MatrixOps, LargeMatrixTranspose2D) {

    int dims[] = {1024, 512};
    Matrix A(dims, 2, 0.0f);
    fill_sequential(A);

    Matrix At = A.transpose2d();

    EXPECT_EQ(At.get_dims_index(0), dims[1]);
    EXPECT_EQ(At.get_dims_index(1), dims[0]);

    for (int i = 0; i < 10; i++) {
        int row = rand() % dims[0];
        int col = rand() % dims[1];

        float original = A.get({row, col});
        float transposed = At.get({col, row});
        EXPECT_FLOAT_EQ(original, transposed);
    }

    EXPECT_FLOAT_EQ(A.get({0,0}), At.get({0,0}));
    EXPECT_FLOAT_EQ(A.get({dims[0]-1, dims[1]-1}), At.get({dims[1]-1, dims[0]-1}));
}

TEST(MatrixOps, SumCols) {

    int dims[] = {2, 3};
    Matrix A(dims, 2, 0.0f);
    fill_sequential(A);

    Matrix row_sums = A.sum_cols();

    EXPECT_EQ(row_sums.get_dims_index(0), 2);
    EXPECT_FLOAT_EQ(row_sums.get({0}), 3);
    EXPECT_FLOAT_EQ(row_sums.get({1}), 12);
}

TEST(MatrixOps, SumRows) {

    int dims[] = {2, 3};
    Matrix A(dims, 2, 0.0f);
    fill_sequential(A);

    Matrix col_sums = A.sum_rows();

    EXPECT_EQ(col_sums.get_dims_index(0), 3);
    EXPECT_FLOAT_EQ(col_sums.get({0}), 3);
    EXPECT_FLOAT_EQ(col_sums.get({1}), 5);
    EXPECT_FLOAT_EQ(col_sums.get({2}), 7);
}

TEST(MatrixOps, SumTotal) {

    int dims[] = {2, 3};
    Matrix A(dims, 2, 0.0f);
    fill_sequential(A);

    float total_sum = A.sum();

    EXPECT_FLOAT_EQ(total_sum, 15);
}

TEST(MatrixOps, SumEdgeCase1x1) {

    int dims[] = {1, 1};
    Matrix A(dims, 2, 0.0f);
    A.set({0,0}, 42);

    Matrix row_sums = A.sum_cols();
    Matrix col_sums = A.sum_rows();
    float total_sum = A.sum();

    EXPECT_FLOAT_EQ(row_sums.get({0}), 42);
    EXPECT_FLOAT_EQ(col_sums.get({0}), 42);
    EXPECT_FLOAT_EQ(total_sum, 42);
}

TEST(MatrixOps, SumZeroMatrix) {

    int dims[] = {3, 4};
    Matrix A(dims, 2, 0.0f);

    Matrix row_sums = A.sum_cols();
    Matrix col_sums = A.sum_rows();
    float total_sum = A.sum();

    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(row_sums.get({i}), 0);
    }
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(col_sums.get({i}), 0);
    }
    EXPECT_FLOAT_EQ(total_sum, 0);
}

TEST(MatrixOps, SumCols_NonDivBy4) {

    int dims[] = {3, 5};
    Matrix A(dims, 2, 0.0f);
    fill_sequential(A);

    Matrix row_sums = A.sum_cols();

    EXPECT_FLOAT_EQ(row_sums.get_dims_index(0), 3);
    EXPECT_FLOAT_EQ(row_sums.get({0}), 10);
    EXPECT_FLOAT_EQ(row_sums.get({1}), 35);
    EXPECT_FLOAT_EQ(row_sums.get({2}), 60);
}

TEST(MatrixOps, SumRows_NonDivBy4) {

    int dims[] = {5, 3};
    Matrix A(dims, 2, 0.0f);
    fill_sequential(A);

    Matrix col_sums = A.sum_rows();

    EXPECT_EQ(col_sums.get_dims_index(0), 3);

    EXPECT_FLOAT_EQ(col_sums.get({0}), 30);
    EXPECT_FLOAT_EQ(col_sums.get({1}), 35);
    EXPECT_FLOAT_EQ(col_sums.get({2}), 40);
}

TEST(MatrixOps, Sum_NonDivBy4) {

    int dims[] = {5, 5};
    Matrix A(dims, 2, 0.0f);
    fill_sequential(A);

    float total_sum = A.sum();

    EXPECT_FLOAT_EQ(total_sum, 300.0f);
}