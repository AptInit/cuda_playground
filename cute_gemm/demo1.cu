#include "cute_gemm.h"
#include <cuda_runtime.h>
#include "util.hpp"
#include "cute/layout.hpp"

template <auto kernelTile, typename TLA, typename TLB, typename TLDst, typename PredFunc>
__device__ static void kernel_block(
    const size_t tileA_idx, const size_t tileB_idx, const size_t tile_idx,
    const TLA& tiledA, const TLB& tiledB, const TLDst& tiledDst,
    PredFunc predicate,
    const float* A, const float* B, const float* C, float* dst) {
    const auto localCoord = cute::make_coord(threadIdx.x, threadIdx.y);
    const auto idxDst = tiledDst(localCoord, tile_idx);
    if (predicate(cute::crd2idx(localCoord, kernelTile), tile_idx)) {
        float acc = C[idxDst];
        for (size_t k=0; k<cute::size<0>(cute::shape<0>(tiledA)); ++k) {
            acc += A[tiledA(cute::make_coord(k, cute::get<1>(localCoord)), tileA_idx)]
                 * B[tiledB(cute::make_coord(cute::get<0>(localCoord), k), tileB_idx)];
        }
        dst[idxDst] = acc;
    }
}

template <dim3 block>
__global__ static void kernel_single_block(
    const size_t m, const size_t n, const size_t k,
    const float* A, const float* B, const float* C,
    float* dst) {
    static_assert(block.z == 1, "2D tiling");
    // All matrices are row-major, fully adopt CuTe's convention now
    auto get_layout = [] __device__ (const auto d0, const auto d1) {
        const auto shape = cute::make_shape(d0, d1);
        return cute::make_layout(shape);
    };
    const auto layoutA = get_layout(k, m);
    const auto layoutB = get_layout(n, k);
    const auto layoutC = get_layout(n, m);
    const auto layoutDst = layoutC;
    // Create block-level tiler for matrices
    constexpr auto blockTile = cute::Shape<cute::C<block.x>, cute::C<block.y>>{}; // tileN, tileM
    const auto tilerA = cute::make_shape(k, cute::get<1>(blockTile));
    const auto tilerB = cute::make_shape(cute::get<0>(blockTile), k);
    // Tile the problem
    const auto tiledA = zipped_divide(layoutA, tilerA);
    const auto tiledB = zipped_divide(layoutB, tilerB);
    const auto tiledDst = zipped_divide(layoutDst, blockTile);
    // Get bounds checker layout for each basis
    auto make_basis = [] __device__ (const auto layout, const auto tiler) {
        // Do stuff recursively in the future
        static_assert(cute::rank(layout) == 2);
        return std::make_tuple(
            cute::zipped_divide(
                cute::make_layout(layout.shape(), cute::make_stride(cute::_1{}, cute::_0{})), tiler),
            cute::zipped_divide(
                cute::make_layout(layout.shape(), cute::make_stride(cute::_0{}, cute::_1{})), tiler));
    };
    auto [boundsDstN, boundsDstM] = make_basis(layoutDst, blockTile);
    auto predicate = [=] __device__ (auto local_idx, auto tile_idx){
        return boundsDstN(local_idx, tile_idx) < cute::size<0>(layoutDst)
            && boundsDstM(local_idx, tile_idx) < cute::size<1>(layoutDst);};
    for (size_t tileIdx = 0; tileIdx < cute::size<1>(tiledDst); ++tileIdx) {
        auto tileCoord = cute::idx2crd(tileIdx, cute::get<1>(tiledDst).shape());
        auto tileCoordA = cute::make_coord(cute::_0{}, cute::get<1>(tileCoord));
        auto tileCoordB = cute::make_coord(cute::get<0>(tileCoord), cute::_0{});
        // Convert 2D tile coordinates to linear indices using crd2idx
        auto tileIdxA = cute::crd2idx(tileCoordA, cute::get<1>(tiledA).shape());
        auto tileIdxB = cute::crd2idx(tileCoordB, cute::get<1>(tiledB).shape());
        kernel_block<blockTile>(
            tileIdxA, tileIdxB, tileIdx,
            tiledA, tiledB, tiledDst, predicate,
            A, B, C, dst);
    }
}

template <dim3 grid, dim3 block>
__global__ static void kernel_grid(
    const size_t m, const size_t n, const size_t k,
    const float* A, const float* B, const float* C,
    float* dst) {
    using namespace cute;
    static_assert(block.z == 1 && grid.z == 1, "2D tiling");
    // All matrices are row-major, fully adopt CuTe's convention now
    auto get_layout = [] __device__ (const auto d0, const auto d1) {
        const auto shape = cute::make_shape(d0, d1);
        return make_layout(shape);
    };
    auto get_basis = [] __device__ (const auto layout, const auto tiler) {
        // Do stuff recursively in the future
        static_assert(cute::rank(layout) == 2);
        return std::make_tuple(
            zipped_divide(
                make_layout(layout.shape(), make_stride(_1{}, _0{})), tiler),
            zipped_divide(
                make_layout(layout.shape(), make_stride(_0{}, _1{})), tiler));
    };
    const auto layoutA = get_layout(k, m);
    const auto layoutB = get_layout(n, k);
    const auto layoutC = get_layout(n, m);
    const auto layoutDst = layoutC;
    // Create tilers for matrices, bottom-up for now, so block-level first
    constexpr auto blockTiler = Shape<cute::C<block.x>, cute::C<block.y>>{}; // tileN, tileM
    const auto blkTilerA = make_shape(k, cute::shape<1>(blockTiler));
    const auto blkTilerB = make_shape(cute::shape<0>(blockTiler), k);
    // Tile the problem
    const auto blkTiledA = zipped_divide(layoutA, blkTilerA);
    const auto blkTiledB = zipped_divide(layoutB, blkTilerB);
    const auto blkTiledDst = zipped_divide(layoutDst, blockTiler);
    // Get bounds checker layout for each basis
    auto [boundsBlkDstN, boundsBlkDstM] = get_basis(layoutDst, blockTiler);
    auto predicate = [=] __device__ (auto local_idx, auto tile_idx){
        return boundsBlkDstN(local_idx, tile_idx) < cute::size<0>(layoutDst)
            && boundsBlkDstM(local_idx, tile_idx) < cute::size<1>(layoutDst);};
    // Grid-level tilers
    const auto gridDomain = make_layout(cute::shape<1>(blkTiledDst));
    constexpr auto gridTiler = Shape<cute::C<grid.x>, cute::C<grid.y>>{}; // blockN, blockM
    const auto grdTiledDst = zipped_divide(gridDomain, gridTiler);
    // Loop through the GEMM problem
    const auto blockCoord = make_coord(blockIdx.x, blockIdx.y);
    for (size_t gridIdx = 0; gridIdx < cute::size<1>(grdTiledDst); ++gridIdx) {
        const auto tileIdx = grdTiledDst(blockCoord, gridIdx);
        if (tileIdx >= cute::size(gridDomain)) { continue; }
        auto tileCoord = idx2crd(tileIdx, cute::get<1>(blkTiledDst).shape());
        auto tileCoordA = make_coord(_0{}, cute::get<1>(tileCoord));
        auto tileCoordB = make_coord(cute::get<0>(tileCoord), _0{});
        // Convert 2D tile coordinates to linear indices using crd2idx
        auto tileIdxA = crd2idx(tileCoordA, cute::get<1>(blkTiledA).shape());
        auto tileIdxB = crd2idx(tileCoordB, cute::get<1>(blkTiledB).shape());
        kernel_block<blockTiler>(
            tileIdxA, tileIdxB, tileIdx,
            blkTiledA, blkTiledB, blkTiledDst, predicate,
            A, B, C, dst);
    }
}

void gemm_f32_row_row_row_row_cuda(
    const size_t m, const size_t n, const size_t k,
    const float* A, const float* B, const float* C,
    float* dst) {
    using namespace util;
    int deviceCount = 0;
    auto error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) {
        log(cudaGetErrorString(error_id));
        abort();
    }
    for (int i = 0; i < deviceCount; ++i) {
        cudaSetDevice(i);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        std::cout<<deviceProp.name<<std::endl;
    }
    cudaSetDevice(0);
    const auto matA = CuPtr(m*k, A);
    const auto matB = CuPtr(k*n, B);
    const auto matC = CuPtr(m*n, C);
    auto matD = CuPtr<float>(m*n);
    // Kernel
    constexpr dim3 grid = {6,8,1};
    constexpr dim3 block = {32,24,1};
    kernel_single_block<block><<<{1,1,1}, block>>>(m,n,k,matA.ptr(), matB.ptr(), matC.ptr(), matD.ptr());
    kernel_grid<grid, block><<<grid, block>>>(m,n,k,matA.ptr(), matB.ptr(), matC.ptr(), matD.ptr());
    cudaMemcpy(dst, matD.ptr(), sizeof(float)*m*n, cudaMemcpyDeviceToHost);
}