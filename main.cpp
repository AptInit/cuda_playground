#include "cute_gemm.h"
#include <iostream>
#include <memory>

int main() {
    std::cout << "It's GEMM time" << std::endl;
    int M = 48, N = 64, K = 65536;
    auto A = std::make_unique<float[]>(M * K);
    auto B = std::make_unique<float[]>(K * N);
    auto C = std::make_unique<float[]>(M * N);
    auto dst = std::make_unique<float[]>(M * N);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            A[i * K + j] = i;
        }
    }
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            B[i * N + j] = j;
        }
    }
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] = i+j;
        }
    }

    //gemm_f32_row_row_row_row(M, N, K, A.get(), B.get(), C.get(), dst.get());
    gemm_f32_row_row_row_row_cuda(M, N, K, A.get(), B.get(), C.get(), dst.get());
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float ref = 0;
            for (int l = 0; l < K; ++l) {
                ref += A[i * K + l] * B[l * N + j];
            }
            ref += C[i * N + j];
            if (std::abs(dst[i * N + j] - ref) > 1e-2) {
                std::cout << "Error at " << i << ", " << j
                    << ": REL DIFF(result/ref-1)=" << (dst[i * N + j]+1e-7f)/(ref+1e-7f)-1 << std::endl;
            } else {
                //std::cout << dst[i * N + j] << "\t";
            }
        }
        //std::cout << std::endl;
    }
    return 0;
}
