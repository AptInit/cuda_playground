#include <cmath>
#include <iostream>
#include <memory>

#include "cute_gemm.h"

bool test_gemm(int M, int N, int K) {
  auto A = std::make_unique<float[]>(M * K);
  auto B = std::make_unique<float[]>(K * N);
  auto C = std::make_unique<float[]>(M * N);
  auto dst = std::make_unique<float[]>(M * N);

  for (int i = 0; i < M; ++i)
    for (int j = 0; j < K; ++j) A[i * K + j] = i;
  for (int i = 0; i < K; ++i)
    for (int j = 0; j < N; ++j) B[i * N + j] = j;
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) C[i * N + j] = i + j;

  gemm_f32_rrrr_cuda_v2(M, N, K, A.get(), B.get(), C.get(), dst.get());

  int errors = 0;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float ref = 0;
      for (int l = 0; l < K; ++l) ref += A[i * K + l] * B[l * N + j];
      ref += C[i * N + j];
      if (std::abs(dst[i * N + j] - ref) > 1e-2f) {
        double relDiff = (dst[i * N + j] + 1e-7f) / (ref + 1e-7f) - 1;
        if (std::abs(relDiff) > 3e-7f) {
          if (errors < 5)
            std::cout << "  Error at " << i << ", " << j
                      << ": REL_DIFF=" << relDiff << std::endl;
          errors++;
        }
      }
    }
  }
  return errors == 0;
}

int main() {
  struct TestCase {
    int M, N, K;
  };
  TestCase tests[] = {
      {192, 384, 65536},  // Original: partial M
      {24, 32, 65536},    // Full tiles
      {1, 32, 65536},     // Extreme partial M
      {48, 31, 65536},    // Partial N
      {23, 17, 65536},    // Partial M and N
      {100, 100, 1000},   // Larger problem
      {1, 1, 96},         // Minimal size
  };

  bool all_pass = true;
  for (const auto& t : tests) {
    std::cout << "Testing M=" << t.M << " N=" << t.N << " K=" << t.K << "... ";
    if (test_gemm(t.M, t.N, t.K)) {
      std::cout << "PASS" << std::endl;
    } else {
      std::cout << "FAIL" << std::endl;
      all_pass = false;
    }
  }
  return all_pass ? 0 : 1;
}
