// #include <cuda_runtime.h>

//Need to get hardware capable for testing.

//A, B, C - pointers in memory, where AxB = C
//A = nxm
//B = mxk
//C = nxk
//Stride A = m
//Stride B = k
//Stride C = k
//Add __global__ tag once I find device compatible with nvcc
void matmul_kernel(float* A, float* B, float* C, int n, int m, int k) {
    //TODO: Tiling, handle matmul of A and B into C
    //Sychronize multiple threads handling matmul for each reigon
}

extern "C" void matmul_cuda(const float* A, const float* B, float* C, int n, int m, int k) {

}