This is an ongoing collection of kernels that explore ways to optimize GEMM kernels in CUDA. It is currently focused on General Matrix Multiplication (GEMM) using NVIDIA's **CuTe** library (part of CUTLASS).

It contains a series of progressive demos:

- **demo0.cu**: Basic row-major GEMM implementation using CuTe.
- **demo1.cu**: Initial CUDA port of the CuTe-based GEMM.
- **demo2.cu**: Optimized CUDA implementation with shared memory tiling.
- **demo3.cu**: Advanced CUDA implementation featuring:
  - Tiled hierarchy (Grid -> Block -> Thread -> Register).
  - Shared memory swizzling for bank conflict avoidance.
  - Multi-level compute loops.

Future work:
- [Done] Auto-tuning
- Vectorized GMEM access
- Double-buffering
- Async-copy
- Lower-precision optimizations such as Tensor Cores, sparcity, micro-scaling formats, etc.

## Documentation

Detailed walkthroughs are provided for the implementations:

- [**CuTe GEMM Walkthrough**](cute_gemm/cute_gemm_walkthrough.md): An introduction to using CuTe for GEMM.
- [**Demo 3 Walkthrough**](cute_gemm/demo3_walkthrough.md): A deep dive into the optimizations used in `demo3.cu`.

## Build Instructions

### Prerequisites

- **CUDA Toolkit** (12.0+ recommended)
- **CMake** (3.24+)
- **C++20** compatible compiler (e.g., GCC 10+, Clang 10+, MSVC 2019+)
- **[CUTLASS CuTe](https://github.com/NVIDIA/cutlass) from NVIDIA** You need to populate the `cute_gemm/include` directory with the CUTLASS headers, DO NOT use the CMakeLists.txt from the CUTLASS repository.

### Building the Project

1. Create a build directory:
   ```bash
   mkdir build
   cd build
   ```

2. Configure with CMake:
   ```bash
   cmake ..
   ```

3. Build:
   ```bash
   cmake --build .
   ```

## Usage

After building, you can run the main test executable:

```bash
./cuda_playground
```

This executable runs a series of test cases (including edge cases like partial tiles and extreme aspect ratios) to verify the correctness of the `demo3` implementation against a CPU reference.

## Project Structure

- `cute_gemm/`: Core library containing the GEMM implementations.
  - `include/`: Public headers and CuTe/CUTLASS dependencies.
  - `demo*.cu`: Sequential optimization steps.
  - `walkthrough*.md`: Detailed technical documentation.
- `main.cpp`: Test harness and validation logic.
- `CMakeLists.txt`: Build configuration.
