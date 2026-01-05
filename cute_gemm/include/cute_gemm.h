//
// Created by jcw on 12/24/25.
//

#pragma once
#include <cstddef>

// demo0.cu
void gemm_f32_row_row_row_row(size_t m, size_t n, size_t k, const float* A,
    const float* B, const float* C, float* dst);

// demo1.cu
void gemm_f32_row_row_row_row_cuda(size_t m, size_t n, size_t k, const float* A,
    const float* B, const float* C, float* dst);

// demo2.cu
void gemm_f32_rrrr_cuda_v2(unsigned m, unsigned n, unsigned k, const float* A,
    const float* B, const float* C, float* dst);

// demo3.cu
void gemm_f32_rrrr_cuda_v3(unsigned m, unsigned n, unsigned k, const float* A,
    const float* B, const float* C, float* dst);
