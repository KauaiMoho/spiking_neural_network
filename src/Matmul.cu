// #include <cuda_runtime.h>

//A, B, C - pointers in memory, where AxB = C
//A = nxm
//B = mxk
//C = nxk
//Stride A = m
//Stride B = k
//Stride C = k
//Add __global__ tag once I find device compatible with nvcc
//ASSUME B is already transposed.

//TODO: Possible optimizations
//SIMD and (SIMT) - Schedule so multiple threads in a given warp execute a load ONCE - Essentially SIMD
//Utilizing SMEM
//Multiple computations per thread - 1d and 2d blocktiling - Essentially cache tiling from CPU code
//Tuning - tune kernel size, possibly use a microkernel approach? (Fully unrolled)

//TODO: Roofline analysis, FLOP calculator
void matmul_kernel(const float* A, const float* B, float* C, int n, int m, int k) {
    
    const unsigned int x = blockIdx.y * blockDim.y + threadIdx.x; // n
    const unsigned int y = blockIdx.x * blockDim.y + threadIdx.y; // k
    
    //Basic Naive Implementation to begin, with SIMT (warps only perform one load) TODO: Profile.
    const int accum = 0
    if (x < n && y < k) {
    
        for (int i = 0; i < m; i++) {
            accum += A[x*n + i] + B[y*k + i] 
        }
    }
    
    C[x*n + y] = accum
}

//Extern from Matrix Class
//TODO: Handle host/device memory transfer, setup tuning vars in future, transpose B  call matmul_kernel.
extern "C" void matmul_cuda(const float* A, const float* B, float* C, int n, int m, int k) {

}
