#include "test_matrix.h"

// Remove: assume_aligned: g++ -O3 -S src/test_alignment.cpp -o matmul_una.s -I include
// Add assume_aligned: g++ -O3 -S src/test_alignment.cpp -o matmul_a.s -I include
// Check diff: diff -u matmul_a.s matmul_una.s | less

float* Matrix::float_size_alloc(size_t size) const {

    size_t a_size = size * sizeof(float);
    size_t remainder = a_size % alignment;
    if (remainder != 0) {
        a_size += alignment - remainder;
    }

    float* arr = (float*) aligned_alloc(alignment, a_size);

    if (arr == nullptr) {
        throw std::runtime_error("Float memory allocation error");
    }

    return arr;
}

std::tuple<int,int,int> Matrix::get_matmul_tile(int n, int m, int k) const {

    //Use loop order to optimize L Cache loading.
    //Use sysctl -a | grep cache to check Apple Silicon Cache Size

    //Can edit later to make this class specific
    constexpr size_t L1_bytes = 64 * 1024;
    constexpr size_t L2_bytes = 4 * 1024 * 1024;
    constexpr int cache_line_size = 128;

    //Choose between L1 and L2 cache based on matrix size
    size_t cache_line_floats = cache_line_size / sizeof(float);
    size_t usable_cache_bytes;
    if (n * m * k * sizeof(float) <= L1_bytes) {
        //Only use max 2/3 of the available bytes in a given cache.
        usable_cache_bytes = L1_bytes / 1.5;
    } else {
        usable_cache_bytes = L2_bytes / 1.5;
    }

    size_t usable_cache_floats = usable_cache_bytes / sizeof(float);

    //Use a heuristic (1/3 of available space for shared dim, assuming near square), and make the other two tile dims porportional to their size
    //T_n*T_m + T_m*T_k + T_n*T_k = usable_cache_floats - need to solve this

    // tile_size^2 * 3 = cache size

    int T_n = static_cast<int>(sqrt((usable_cache_floats * n) / (3.0 * k)));
    int T_m = static_cast<int>(sqrt(usable_cache_floats / 3.0));
    int T_k = static_cast<int>(sqrt((usable_cache_floats * k) / (3.0 * n)));

    //Now we round down to cache line size. Could use bit masking since cache line power of 2 (originally i did this, but changed back for clarity): mat_tile & ~(cache_line_floats - 1)
    //We do this to make sure all cache line data loads are perfectly used, and round down to make sure we do not exceed usable cache size.
    T_n -= T_n%cache_line_floats;
    T_m -= T_m%cache_line_floats;
    T_k -= T_k%cache_line_floats;
    
    if (T_n == 0) {
        //If tile size is small, stick with cache line float size.
        T_n = cache_line_floats;
    }

    if (T_m == 0) {
        T_m = cache_line_floats;
    }

    if (T_k == 0) {
        T_k = cache_line_floats;
    }


    //Now we will only work with sub matrices of size mat_tile, which is good as it will allow the matmul code to always pull/write straight from CPU cache, avoiding slower memory access.
    return  std::make_tuple(T_n, T_m, T_k);
}

void Matrix::matmul_cpu(const float* A, const float* B, float* C, int n, int m, int k) const {

    //A = nxm
    //B = mxk
    //C = nxk
    //Stride A = m
    //Stride B = k
    //Stride C = k
    //Assume matrix dimensions near square for cache optimization simplicity.
   
    std::tuple<int, int, int> tile_data = get_matmul_tile(n, m, k);
    int T_n = std::get<0>(tile_data);
    int T_m = std::get<1>(tile_data); //shared dim
    int T_k = std::get<2>(tile_data);

    float* B_t = float_size_alloc(m * k);
    
    simd_transpose(B, B_t, m, k);
    for (size_t ic = 0; ic < n; ic += T_n) {
        for (size_t lc = 0; lc < k; lc += T_k){
            size_t iE = std::min(ic + T_n, static_cast<size_t>(n));
            for (size_t i = ic; i < iE; ++i){
                size_t lE = std::min(lc + T_k, static_cast<size_t>(k));
                for (size_t l = lc; l < lE; ++l) {
                    float sum = 0;
                    float32x4_t acc = vdupq_n_f32(0.0f);
                    for (size_t jc = 0; jc < m; jc += T_m) {
                        size_t jE =  std::min(jc + T_m, static_cast<size_t>(m));
                        const float* ptrA = assume_aligned(&A[i*m + jc]);
                        const float* ptrB = assume_aligned(&B_t[l*m + jc]);
                        for (size_t j = jc; j + 3 < jE; j += 4) {
                            float32x4_t a = vld1q_f32(assume_aligned(ptrA));
                            float32x4_t b = vld1q_f32(assume_aligned(ptrB));
                            ptrA += 4;
                            ptrB += 4;
                            acc = vaddq_f32(acc, vmulq_f32(a, b));
                        }
                        for (size_t j = 0; j < (jE%4); ++j) {
                            sum += (*ptrA) * (*ptrB);
                            ptrA += 1;
                            ptrB += 1;
                        }
                    }
                    sum += vaddvq_f32(acc);
                    C[i*k + l] = sum;
                }   
            }
        }
    }
    free(B_t);
}

void Matrix::simd_transpose(const float* A, float* B, int n, int m, int z, const int* dists_new) const {

    //We choose to repeat code rather than make a temp dists var so that the original dists_new can stay const (cannot free if 
    //we use const temp_dists_new).

    if (!dists_new) {
        size_t offset = n*m*z;
        //Tile for same reasons as matmul (minimize cache misses)
        for (size_t ic = 0; ic + tile <= n; ic += tile) {
            for (size_t jc = 0; jc + tile <= m; jc += tile) {
                for (size_t i = ic; i < ic+tile; i += 4) {
                    for (size_t j = jc; j < jc+tile; j += 4) {

                        //Does 4x4 sections, and a scalar cleanup.
                        //Load 16 elements from A to tranpose into B
                        // a = [a0 a1 a2 a3]
                        // b = [b0 b1 b2 b3]
                        // c = [c0 c1 c2 c3]
                        // d = [d0 d1 d2 d3]

                        
                        float32x4_t a = vld1q_f32(assume_aligned(&A[offset + i*m + j]));
                        float32x4_t b = vld1q_f32(assume_aligned(&A[offset + (i + 1)*m + j]));
                        float32x4_t c = vld1q_f32(assume_aligned(&A[offset + (i + 2)*m + j]));
                        float32x4_t d = vld1q_f32(assume_aligned(&A[offset + (i + 3)*m + j]));

                        //Transpose halves (swap even and odd lanes)
                        //[a0 a1 a2 a3]      [a0 b0 a2 b2]
                        //[b0 b1 b2 b3]  →   [a1 b1 a3 b3]
                        float32x4x2_t p0 = vtrnq_f32(a, b);
                        float32x4x2_t p1 = vtrnq_f32(c, d);

                        //Combine halves

                        // low(p0[0]) = [a0 b0]
                        // low(p1[0]) = [c0 d0]
                        // → r0 = [a0 b0 c0 d0]
                        float32x4_t r0 = vcombine_f32(vget_low_f32(p0.val[0]), vget_low_f32(p1.val[0]));
                        float32x4_t r1 = vcombine_f32(vget_low_f32(p0.val[1]), vget_low_f32(p1.val[1]));

                        // high(p0[0]) = [a2 b2]
                        // high(p1[0]) = [c2 d2]
                        // → r2 = [a2 b2 c2 d2]
                        float32x4_t r2 = vcombine_f32(vget_high_f32(p0.val[0]), vget_high_f32(p1.val[0]));
                        float32x4_t r3 = vcombine_f32(vget_high_f32(p0.val[1]), vget_high_f32(p1.val[1]));

                        //Store into B (no alignment checks)
                        vst1q_f32(&B[j*n + i], r0);
                        vst1q_f32(&B[(j + 1)*n + i], r1);
                        vst1q_f32(&B[(j + 2)*n + i], r2);
                        vst1q_f32(&B[(j + 3)*n + i], r3);
                    }
                }
            }
        }

        //Scalar Clean up what was missed by tiling

        //Handles leftover rows - Bottom Rectangle
        for (size_t i = n-(n%tile); i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                B[j*n + i] = A[offset + i*m + j];
            }
        }

        // Corner:
        // i >= n-(n%tile)
        // j >= m-(m%tile)

        //Handle leftover colouns (ignoring final few rows overlapping with above loop)
        //Basically the top right rectangle 
        for (size_t i = 0; i < n-(n%tile); ++i) {
            for (size_t j = m-(m%tile); j < m; ++j) {
                B[j*n + i] = A[offset + i*m + j];
            }
        }

    } else {
        
        for (size_t ic = 0; ic + tile <= n; ic += tile) {
            for (size_t jc = 0; jc + tile <= m; jc += tile) {
                for (size_t i = ic; i < ic+tile; i += 4) {
                    for (size_t j = jc; j < jc+tile; j += 4) {
                        float32x4_t a = vld1q_f32(assume_aligned(&A[z*dists_new[0] + i*dists_new[1] + j*dists_new[2]]));
                        float32x4_t b = vld1q_f32(assume_aligned(&A[z*dists_new[0] + (i + 1)*dists_new[1] + j*dists_new[2]]));
                        float32x4_t c = vld1q_f32(assume_aligned(&A[z*dists_new[0] + (i + 2)*dists_new[1] + j*dists_new[2]]));
                        float32x4_t d = vld1q_f32(assume_aligned(&A[z*dists_new[0] + (i + 3)*dists_new[1] + j*dists_new[2]]));
                        
                        float32x4x2_t p0 = vtrnq_f32(a, b);
                        float32x4x2_t p1 = vtrnq_f32(c, d);

                        float32x4_t r0 = vcombine_f32(vget_low_f32(p0.val[0]), vget_low_f32(p1.val[0]));
                        float32x4_t r1 = vcombine_f32(vget_low_f32(p0.val[1]), vget_low_f32(p1.val[1]));
                        float32x4_t r2 = vcombine_f32(vget_high_f32(p0.val[0]), vget_high_f32(p1.val[0]));
                        float32x4_t r3 = vcombine_f32(vget_high_f32(p0.val[1]), vget_high_f32(p1.val[1]));

                        vst1q_f32(&B[j*n + i], r0);
                        vst1q_f32(&B[(j + 1)*n + i], r1);
                        vst1q_f32(&B[(j + 2)*n + i], r2);
                        vst1q_f32(&B[(j + 3)*n + i], r3);
                    }
                }
            }
        }

        for (size_t i = n-(n%tile); i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                B[j*n + i] = A[z*dists_new[0] + i*dists_new[1] + j*dists_new[2]];
            }
        }

        for (size_t i = 0; i < n-(n%tile); ++i) {
            for (size_t j = m-(m%tile); j < m; ++j) {
                B[j*n + i] = A[z*dists_new[0] + i*dists_new[1] + j*dists_new[2]];
            }
        }
    }
}