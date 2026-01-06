//
// Created by jcw on 12/26/25.
//

#pragma once

#include <cuda_runtime.h>

#include <iostream>
#include <numeric>
#include <optional>
#include <source_location>

namespace util {
template <dim3 vec>
struct SDim3 {
  using X = cute::Int<vec.x>;
  using Y = cute::Int<vec.y>;
  using Z = cute::Int<vec.z>;
  X x;
  Y y;
  Z z;
};

// The recursive loader function
// T: The element type (float)
// N: The remaining number of elements to load
template <int N, typename T>
__device__ __forceinline__ void load_vectorized(T* dst, const T* src) {
  static_assert(sizeof(T) == 4);
  if constexpr (N >= 4) {
    // Load 128 bits (float4)
    using VecType = float4;

    *reinterpret_cast<VecType*>(dst) = *reinterpret_cast<const VecType*>(src);

    // Recurse for the rest
    load_vectorized<N - 4>(dst + 4, src + 4);
  } else if constexpr (N >= 2) {
    // Load 64 bits (float2)
    using VecType = float2;

    *reinterpret_cast<VecType*>(dst) = *reinterpret_cast<const VecType*>(src);

    load_vectorized<N - 2>(dst + 2, src + 2);
  } else if constexpr (N == 1) {
    // Load 32 bits (scalar)
    *dst = *src;
  }
}
template <std::integral T>
__host__ __device__ constexpr T floor(const T num, const T factor) {
  return num / factor * factor;
}
template <std::integral T>
__host__ __device__ constexpr T ceiling(const T num, const T factor) {
  return floor(num + factor - 1, factor);
}
template <std::integral T>
__host__ __device__ constexpr std::tuple<T, T> lcm_gcd(const T a, const T b) {
  return {std::lcm(a, b), std::gcd(a, b)};
}
template <std::integral T>
__host__ __device__ consteval T static_logk(T num, const T base) {
  T result = 0;
  while (num > base) {
    num /= base;
    result += 1;
  }
  return result + (num == base);
}
template <std::integral T>
__host__ __device__ consteval T static_powk(T num, const T base) {
  T result = 1;
  while (num > 0) {
    num -= 1;
    result *= base;
  }
  return result;
}
inline void log(const std::optional<std::string_view>& prompt = std::nullopt,
    const std::source_location& location = std::source_location::current()) {
  std::cout << "" << location.file_name() << ':' << location.line() << ' '
            << (prompt.has_value() ? prompt.value() : "") << std::endl;
}

inline void check_cuda(const cudaError_t &ret_code,
  const std::source_location& location = std::source_location::current()) {
  if (ret_code != cudaSuccess) {
    log(cudaGetErrorString(ret_code), location);
    abort();
  }
}

template <typename T>
class CuPtr {
  static_assert(std::is_trivially_copyable_v<T>);
  T* data_ = nullptr;
  size_t size_ = 0;

 public:
  CuPtr(const size_t size, const T* host) : size_(size) {
    const auto ret = cudaMallocManaged(&data_, size * sizeof(T));
    if (ret != cudaSuccess) {
      log(cudaGetErrorString(ret));
      abort();
    }
    if (host != nullptr) {
      cudaMemcpy(data_, host, size * sizeof(T), cudaMemcpyHostToDevice);
    }
  }
  explicit CuPtr(const size_t size) : CuPtr(size, nullptr) {}
  ~CuPtr() {
    if (data_) {
      const auto ret = cudaFree(data_);
      if (ret != cudaSuccess) {
        log(cudaGetErrorString(ret));
        abort();
      }
      data_ = nullptr;
      size_ = 0;
    }
  }
  CuPtr(const CuPtr&) = delete;
  CuPtr& operator=(const CuPtr&) = delete;
  CuPtr& operator=(CuPtr&& other) noexcept {
    if (this != &other) {
      std::swap(data_, other.data_);
      std::swap(size_, other.size_);
    }
    return *this;
  }
  CuPtr(CuPtr&& other) noexcept { *this = std::move(other); }
  const T* ptr() const { return data_; }

  T* ptr() { return data_; }
};
}  // namespace util
