//
// Created by jcw on 12/26/25.
//

#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <optional>
#include <source_location>
#include <thread>

namespace util
{
    inline void log(const std::optional<std::string_view> &prompt = std::nullopt,
        const std::source_location& location = std::source_location::current()){
        std::cout << ""
                  << location.file_name() << ':'
                  << location.line() << ' '
                  << (prompt.has_value() ? prompt.value() : "") << std::endl;
    }
    template <typename T>
    class CuPtr {
        static_assert(std::is_trivially_copyable_v<T>);
        T* data_ = nullptr;
        size_t size_ = 0;
    public:
        CuPtr(const size_t size, const T* host): size_(size) {
            const auto ret = cudaMallocManaged(&data_, size * sizeof(T));
            if (ret != cudaSuccess) {
                log(cudaGetErrorString(ret));
                abort();
            }
            if (host != nullptr) {
                cudaMemcpy(data_, host, size * sizeof(T), cudaMemcpyHostToDevice);
            }
        }
        explicit CuPtr(const size_t size): CuPtr(size, nullptr) {}
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
        CuPtr(CuPtr&& other) noexcept {
            *this = std::move(other);
        }
        const T* ptr() const {
            return data_;
        }

        T* ptr() {
            return data_;
        }
    };
}
