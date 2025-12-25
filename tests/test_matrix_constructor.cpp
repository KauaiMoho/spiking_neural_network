#include <gtest/gtest.h>
#include "Matrix.h"

TEST(MatrixConstructor, FillWithValue) {

    int dims[] = {2, 2};
    Matrix M(dims, 2, 5.0f);

    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(M.get_index(i), 5.0f);
    }
}

TEST(MatrixConstructor, CopyData) {

    int dims[] = {2, 2};
    float data[] = {1, 2, 3, 4};

    Matrix M(dims, 2, data);
    data[0] = 99;

    EXPECT_FLOAT_EQ(M.get({0, 0}), 1.0f);
}

TEST(MatrixConstructor, RandomSeed) {

    int dims[] = {2, 2};

    Matrix A(dims, 2, (unsigned int)123);
    Matrix B(dims, 2, (unsigned int)123);

    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(A.get_index(i), B.get_index(i));
    }
}

TEST(MatrixConstructor, CopyConstructor) {

    int dims[] = {2, 2};
    float data[] = {1, 2, 3, 4};

    Matrix A(dims, 2, data);
    Matrix B(A);

    data[0] = 99;

    EXPECT_FLOAT_EQ(B.get({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(B.get({1, 1}), 4.0f);
}

TEST(MatrixConstructor, CopyAssignment) {

    int dims[] = {2, 2};
    float data1[] = {1, 2, 3, 4};
    float data2[] = {5, 6, 7, 8};

    Matrix A(dims, 2, data1);
    Matrix B(dims, 2, data2);

    B = A;

    EXPECT_FLOAT_EQ(B.get({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(B.get({1, 1}), 4.0f);
}

TEST(MatrixConstructor, SelfAssignment) {

    int dims[] = {2, 2};
    float data[] = {1, 2, 3, 4};

    Matrix M(dims, 2, data);
    M = M;

    EXPECT_FLOAT_EQ(M.get({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(M.get({1, 1}), 4.0f);
}

TEST(MatrixConstructor, MoveConstructor) {

    int dims[] = {2, 2};
    float data[] = {1, 2, 3, 4};

    Matrix A(dims, 2, data);
    Matrix B(std::move(A));

    EXPECT_FLOAT_EQ(B.get({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(B.get({1, 1}), 4.0f);
}

TEST(MatrixConstructor, MoveAssignment) {

    int dims[] = {2, 2};
    float data1[] = {1, 2, 3, 4};
    float data2[] = {5, 6, 7, 8};

    Matrix A(dims, 2, data1);
    Matrix B(dims, 2, data2);

    B = std::move(A);

    EXPECT_FLOAT_EQ(B.get({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(B.get({1, 1}), 4.0f);
}