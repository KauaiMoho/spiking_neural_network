// #include <cuda_runtime.h>

//A, B, C - pointers in memory, where AxB = C
//A = nxm
//B = mxk
//C = nxk
//Stride A = m
//Stride B = k
//Stride C = k
//Add __global__ tag once I find device compatible with nvcc
void matmul_kernel(float* A, float* B, float* C, int n, int m, int k) {
    //Note that CUDA kernels are written in thread perspective
    // const unsigned int x = blockIdx.y * blockDim.y + threadIdx.y;
    // const unsigned int y = blockIdx.x * blockDim.y + threadIdx.x;
}

//Extern from Matrix Class
extern "C" void matmul_cuda(const float* A, const float* B, float* C, int n, int m, int k) {

}