#include "../include/Matrix.h" // Assuming this contains the Matrix class definition
#include <iostream>
#include <vector>
#include <cmath> // For std::abs
#include <stdexcept>

// Helper function to check if two floats are approximately equal
bool nearly_equal(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) <= epsilon;
}

// Helper function to check if two matrices are approximately equal
bool matrix_nearly_equal(const Matrix& m1, const Matrix& m2, float epsilon = 1e-5f) {
    if (m1.get_dim_len() != m2.get_dim_len()) {
        std::cerr << "Mismatch in dim_len: " << m1.get_dim_len() << " vs " << m2.get_dim_len() << std::endl;
        return false;
    }
    for (int i = 0; i < m1.get_dim_len(); ++i) {
        if (m1.get_dims_index(i) != m2.get_dims_index(i)) {
            std::cerr << "Mismatch in dims[" << i << "]: " << m1.get_dims_index(i) << " vs " << m2.get_dims_index(i) << std::endl;
            return false;
        }
    }
    
    // Assuming a method to get the linear data length is available or calculating it:
    size_t data_len = 1;
    for (int i = 0; i < m1.get_dim_len(); ++i) {
        data_len *= m1.get_dims_index(i);
    }

    const float* data1 = m1.get_data();
    const float* data2 = m2.get_data();

    for (size_t i = 0; i < data_len; ++i) {
        if (!nearly_equal(data1[i], data2[i], epsilon)) {
            std::cerr << "Mismatch at index " << i << ": " << data1[i] << " vs " << data2[i] << std::endl;
            return false;
        }
    }
    return true;
}

// Helper to run a test and report success/failure
void run_test(const std::string& name, bool (*test_func)()) {
    std::cout << "--- Running Test: " << name << " ---" << std::endl;
    if (test_func()) {
        std::cout << "[PASS] " << name << std::endl;
    } else {
        std::cerr << "[FAIL] " << name << std::endl;
    }
    std::cout << std::endl;
}

// -------------------------------------------------------------------
// Test Cases based on `dim_len` and `other.get_dim_len()`
// -------------------------------------------------------------------

// 1. Dot Product (1x1 . 1x1)
bool test_dot_product_success() {
    try {
        int dim1[] = {3}; // 3 elements (a vector)
        int dim2[] = {3}; // 3 elements (a vector)

        float dataA[] = {1.0f, 2.0f, 3.0f};
        float dataB[] = {4.0f, 5.0f, 6.0f};

        Matrix A(dim1, 1, dataA);
        Matrix B(dim2, 1, dataB);

        // Expected result: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32.0
        Matrix C = A.matmul(B);

        int expected_dim[] = {1};
        Matrix expected_C(expected_dim, 1, 32.0f);
        
        return matrix_nearly_equal(C, expected_C);
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return false;
    }
}

bool test_dot_product_fail_dim() {
    try {
        int dim1[] = {3};
        int dim2[] = {4};

        float dataA[] = {1.0f, 2.0f, 3.0f};
        float dataB[] = {4.0f, 5.0f, 6.0f, 7.0f};

        Matrix A(dim1, 1, dataA);
        Matrix B(dim2, 1, dataB);

        A.matmul(B); // Should throw
        return false; // Test failed if no exception thrown
    } catch (const std::invalid_argument& e) {
        // Expected exception caught
        std::cout << "Caught expected exception: " << e.what() << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Caught unexpected exception: " << e.what() << std::endl;
        return false;
    }
}

// 2. Vector Product (Matrix X Vector) (2x1 . 1x1) -> n x m X m x 1 = n x 1
bool test_matrix_vector_product_success() {
    try {
        int dimA[] = {2, 3}; // 2x3 matrix
        int dimB[] = {3};    // 3x1 vector

        // A = [[1, 2, 3], [4, 5, 6]]
        float dataA[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        // B = [7, 8, 9]
        float dataB[] = {7.0f, 8.0f, 9.0f};

        Matrix A(dimA, 2, dataA);
        Matrix B(dimB, 1, dataB);

        // Expected result:
        // C[0] = 1*7 + 2*8 + 3*9 = 7 + 16 + 27 = 50.0
        // C[1] = 4*7 + 5*8 + 6*9 = 28 + 40 + 54 = 122.0
        float expected_data[] = {50.0f, 122.0f};
        int expected_dim[] = {2}; // 2x1 vector
        Matrix expected_C(expected_dim, 1, expected_data);

        Matrix C = A.matmul(B);
        return matrix_nearly_equal(C, expected_C);
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return false;
    }
}

bool test_matrix_vector_product_fail_dim() {
    try {
        int dimA[] = {2, 3}; // 2x3 matrix
        int dimB[] = {2};    // 2x1 vector (mismatch: m=3 != 2)

        float dataA[6] = {1.0f};
        float dataB[2] = {1.0f};

        Matrix A(dimA, 2, dataA);
        Matrix B(dimB, 1, dataB);

        A.matmul(B); // Should throw
        return false;
    } catch (const std::invalid_argument& e) {
        std::cout << "Caught expected exception: " << e.what() << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Caught unexpected exception: " << e.what() << std::endl;
        return false;
    }
}

// 3. Vector Product (Vector X Matrix) (1x2 . 2x2) -> 1 x m X m x n = 1 x n
bool test_vector_matrix_product_success() {
    try {
        int dimA[] = {3};    // 1x3 vector (passed as a 1D vector of length 3)
        int dimB[] = {3, 2}; // 3x2 matrix

        // A = [1, 2, 3]
        float dataA[] = {1.0f, 2.0f, 3.0f};
        // B = [[4, 5], [6, 7], [8, 9]]
        float dataB[] = {4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

        Matrix A(dimA, 1, dataA);
        Matrix B(dimB, 2, dataB);

        // Expected result (1x2 vector):
        // C[0] = 1*4 + 2*6 + 3*8 = 4 + 12 + 24 = 40.0
        // C[1] = 1*5 + 2*7 + 3*9 = 5 + 14 + 27 = 46.0
        float expected_data[] = {40.0f, 46.0f};
        int expected_dim[] = {2}; // 1x2 vector (represented as a 1D vector of length 2 by the function)
        Matrix expected_C(expected_dim, 1, expected_data);

        Matrix C = A.matmul(B);
        return matrix_nearly_equal(C, expected_C);
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return false;
    }
}

bool test_vector_matrix_product_fail_dim() {
    try {
        int dimA[] = {3};    // 1x3 vector
        int dimB[] = {2, 2}; // 2x2 matrix (mismatch: 3 != 2)

        float dataA[3] = {1.0f};
        float dataB[4] = {1.0f};

        Matrix A(dimA, 1, dataA);
        Matrix B(dimB, 2, dataB);

        A.matmul(B); // Should throw
        return false;
    } catch (const std::invalid_argument& e) {
        std::cout << "Caught expected exception: " << e.what() << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Caught unexpected exception: " << e.what() << std::endl;
        return false;
    }
}

// 4. Matrix Multiplication (2x2 . 2x2)
// NOTE: This relies on an external matmul_cpu/matmul_cuda function, so the test checks the wrapper logic and output size.
bool test_matrix_matrix_product_success() {
    // We assume matmul_cpu/matmul_cuda works correctly for this test.
    // We primarily check if the wrapper logic sets up the correct output dimensions.
    try {
        int dimA[] = {2, 2}; // 2x2
        int dimB[] = {2, 2}; // 2x2

        // A = [[1, 2], [3, 4]]
        float dataA[] = {1.0f, 2.0f, 3.0f, 4.0f};
        // B = [[5, 6], [7, 8]]
        float dataB[] = {5.0f, 6.0f, 7.0f, 8.0f};

        Matrix A(dimA, 2, dataA);
        Matrix B(dimB, 2, dataB);

        // Expected result (2x2):
        // C[0,0] = 1*5 + 2*7 = 5 + 14 = 19.0
        // C[0,1] = 1*6 + 2*8 = 6 + 16 = 22.0
        // C[1,0] = 3*5 + 4*7 = 15 + 28 = 43.0
        // C[1,1] = 3*6 + 4*8 = 18 + 32 = 50.0
        float expected_data[] = {19.0f, 22.0f, 43.0f, 50.0f};
        int expected_dim[] = {2, 2};
        Matrix expected_C(expected_dim, 2, expected_data);

        Matrix C = A.matmul(B);
        return matrix_nearly_equal(C, expected_C);
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return false;
    }
}

bool test_matrix_matrix_product_fail_dim() {
    try {
        int dimA[] = {2, 3}; // 2x3
        int dimB[] = {2, 2}; // 2x2 (mismatch: 3 != 2)

        float dataA[6] = {1.0f};
        float dataB[4] = {1.0f};

        Matrix A(dimA, 2, dataA);
        Matrix B(dimB, 2, dataB);

        A.matmul(B); // Should throw
        return false;
    } catch (const std::invalid_argument& e) {
        std::cout << "Caught expected exception: " << e.what() << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Caught unexpected exception: " << e.what() << std::endl;
        return false;
    }
}

// 5. Batched Matrix Multiplication (>= 2x2 . >= 2x2)
// This case is complex and relies on correct implementation of broadcast, reshape, and batched matmul.
// We'll test the output shape and broadcasting logic.
bool test_batched_matmul_with_broadcasting() {
    // Test: (1, 2, 3) X (4, 3, 2) -> Broadcating from (1, 2, 3) to (4, 2, 3)
    // Result shape should be (4, 2, 2)
    // NOTE: This test will be heavily dependent on the *exact* implementation of broadcast and the batched CPU/CUDA function.
    // For a basic sanity check, we verify the output dimensions.
    // The provided function logic is complex and assumes many external functions.
    
    // Since we cannot run the actual complex logic, we focus on checking the expected outcome
    // of the dimension calculation part of the batched matmul.

    // A = 1x2x3
    int dimA[] = {1, 2, 3}; 
    // B = 4x3x2
    int dimB[] = {4, 3, 2}; 
    
    try {
        // Dummy data for construction
        Matrix A(dimA, 3, 2.0f); 
        Matrix B(dimB, 3, 1.0f);
        
        Matrix C = A.matmul(B);
        
        // Expected result shape is: 
        // Broadcasted batch dims: max(1, 4) = 4, max(2, 3) fails for the general dims, 
        // but the logic only checks up to 'dim_len - 3' and then sets the last two.
        // The logic seems to be:
        // Batch Dims: max(dims[i], other.dims[i]) if one is 1, else must match.
        // A's batch dim: {1}
        // B's batch dims: {4}
        // Broadcasted batch dim: {4}
        // Output shape: {4, M, N} where M=A[..., -2] and N=B[..., -1]
        // M = 2, N = 2
        // Expected shape: {4, 2, 2}
        
        int expected_dim_len = 3;
        int expected_dims[] = {4, 2, 2};
        
        if (C.get_dim_len() != expected_dim_len) return false;
        for (int i = 0; i < expected_dim_len; ++i) {
            if (C.get_dims_index(i) != expected_dims[i]) return false;
        }
        
        // Additional check: The matmul of a 1x2x3 (all 2.0) and 4x3x2 (all 1.0)
        // should result in a 4x2x2 matrix where each element is 3 * 2 * 1 = 6.0
        // (inner dim 3, A value 2, B value 1).
        
        float expected_val = 3.0f * 2.0f * 1.0f; // 6.0
        size_t total_elements = 4 * 2 * 2;

        const float* c_data = C.get_data();
        for (size_t i = 0; i < total_elements; ++i) {
            if (!nearly_equal(c_data[i], expected_val)) return false;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return false;
    }
}

bool test_batched_matmul_fail_dim() {
    try {
        // A = 2x2x3 (inner dim 3)
        int dimA[] = {2, 2, 3};
        // B = 2x4x2 (inner dim 4 != 3)
        int dimB[] = {2, 4, 2}; 

        Matrix A(dimA, 3, 1.0f);
        Matrix B(dimB, 3, 1.0f);

        A.matmul(B); // Should throw
        return false;
    } catch (const std::invalid_argument& e) {
        std::cout << "Caught expected exception: " << e.what() << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Caught unexpected exception: " << e.what() << std::endl;
        return false;
    }
}


// Main test runner
int main() {
    // Set environment flag if necessary
    Matrix::set_CUDA(false); 
    
    // 1. Dot Product Tests
    run_test("Dot Product Success", test_dot_product_success);
    run_test("Dot Product Fail Dimensions", test_dot_product_fail_dim);
    
    // 2. Matrix x Vector Product Tests
    run_test("Matrix x Vector Product Success", test_matrix_vector_product_success);
    run_test("Matrix x Vector Product Fail Dimensions", test_matrix_vector_product_fail_dim);
    
    // 3. Vector x Matrix Product Tests
    run_test("Vector x Matrix Product Success", test_vector_matrix_product_success);
    run_test("Vector x Matrix Product Fail Dimensions", test_vector_matrix_product_fail_dim);
    
    // 4. Matrix x Matrix Product Tests
    run_test("Matrix x Matrix Product Success", test_matrix_matrix_product_success);
    run_test("Matrix x Matrix Product Fail Dimensions", test_matrix_matrix_product_fail_dim);
    
    // 5. Batched Matrix Multiplication Tests
    run_test("Batched Matmul with Broadcasting Success (1x2x3 X 4x3x2)", test_batched_matmul_with_broadcasting);
    run_test("Batched Matmul Fail Dimensions (Inner Mismatch)", test_batched_matmul_fail_dim);

    return 0;
}