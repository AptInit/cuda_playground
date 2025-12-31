#include "cute_gemm.h"
#include <cuda_runtime.h>
#include "util.hpp"
#include "cute/layout.hpp"

template <auto kernelTile, unsigned tileK, typename TLA, typename TLB, typename TLDst, typename PredFunc>
__device__ static void kernel_block(
    const unsigned tileA_idx, const unsigned tileB_idx, const unsigned tile_idx,
    const TLA& tiledA, const TLB& tiledB, const TLDst& tiledDst,
    PredFunc predicate,
    const float* A, const float* B, const float* C, float* dst) {
    constexpr unsigned tileN = cute::size<0>(kernelTile);
    constexpr unsigned tileM = cute::size<1>(kernelTile);
    // (tileM, tileKHi), tileKLo
    __shared__ float scratchA_SMEM[tileM*tileK/32][32];
    // tileK, tileN
    __shared__ float scratchB_SMEM[tileK*tileN/32][32];
    const auto localCoord = cute::make_coord(threadIdx.x, threadIdx.y);
    const auto idxDst = tiledDst(localCoord, tile_idx);
    const auto [isValidN, isValidM] = predicate(cute::crd2idx(localCoord, kernelTile), tile_idx);
    const bool isValid = isValidN && isValidM;
    float acc = isValid ? C[idxDst] : 0.0f;
    const unsigned dimK = cute::size<0>(cute::shape<0>(tiledA));
    // Don't want to spam layout algebra here
    const auto cntK = dimK/tileK;
    for (unsigned tileKIdx=0; tileKIdx<cntK;++tileKIdx) {
        const auto kOffset = tileKIdx*tileK;
        // Load scratch
        __syncthreads();
        for (unsigned i=0; i<tileK*tileM; i+=tileM*tileN) {
            scratchA_SMEM[threadIdx.y*tileK/32+i/tileM/32][threadIdx.x] =
                isValidM ? A[tiledA(cute::make_coord(kOffset+i/tileM+threadIdx.x, cute::get<1>(localCoord)), tileA_idx)] : 0.0f;
        }
        for (unsigned i=0; i<tileK*tileN; i+=tileM*tileN) {
            scratchB_SMEM[threadIdx.y+i/tileN][threadIdx.x] =
                isValidN ? B[tiledB(cute::make_coord(cute::get<0>(localCoord), kOffset+i/tileN+threadIdx.y), tileB_idx)] : 0.0f;
        }
        __syncthreads();
        // for (unsigned kBase=0; kBase<tileK; kBase+=32) {
        //     const auto tempA = scratchA_SMEM[threadIdx.y*tileK/32+kBase/32][threadIdx.x];
        //     __syncwarp();
        //     for (unsigned kIdx=0; kIdx<32;++kIdx) {
        //         acc += __shfl_sync(0xFFFFFFFF, tempA, kIdx) * scratchB_SMEM[(kIdx+kBase)*(tileN/32)+threadIdx.x/32][threadIdx.x%32];
        //     }
        // }
        for (unsigned k=0; k<tileK; ++k) {
            acc += scratchA_SMEM[threadIdx.y*tileK/32+k/32][k%32] * scratchB_SMEM[k*(tileN/32)+threadIdx.x/32][threadIdx.x%32];
        }
    }
    if (isValid) {
        for (unsigned k=cntK*tileK; k<dimK; ++k) {
            acc += A[tiledA(cute::make_coord(k, cute::get<1>(localCoord)), tileA_idx)]
                 * B[tiledB(cute::make_coord(cute::get<0>(localCoord), k), tileB_idx)];
        }
        dst[idxDst] = acc;
    }
}

template <unsigned grid_x, unsigned grid_y, unsigned grid_z, unsigned block_x, unsigned block_y, unsigned block_z>
__global__ static void kernel_grid_v2(
    const unsigned m, const unsigned n, const unsigned k,
    const float* A, const float* B, const float* C,
    float* dst) {
    using namespace cute;
    constexpr dim3 grid{grid_x, grid_y, grid_z}, block{block_x, block_y, block_z};
    static_assert(block.z == 1 && grid.z == 1, "2D tiling");
    // All matrices are row-major, but CuTe uses colexicographical order, follow CuTe's convention
    auto get_layout = [] __device__ (const auto d0, const auto d1) {
        const auto shape = cute::make_shape(d0, d1);
        return make_layout(shape);
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
    const auto boundsBlkDstN = zipped_divide(
                make_layout(layoutDst.shape(), make_stride(_1{}, _0{})), blockTiler);
    const auto boundsBlkDstM = zipped_divide(
                make_layout(layoutDst.shape(), make_stride(_0{}, _1{})), blockTiler);
    auto predicate = [=] __device__ (auto local_idx, auto tile_idx){
        return std::make_tuple(boundsBlkDstN(local_idx, tile_idx) < cute::size<0>(layoutDst),
                               boundsBlkDstM(local_idx, tile_idx) < cute::size<1>(layoutDst));
    };
    // Grid-level tilers
    const auto gridDomain = make_layout(cute::shape<1>(blkTiledDst));
    constexpr auto gridTiler = Shape<cute::C<grid.x>, cute::C<grid.y>>{}; // blockN, blockM
    const auto grdTiledDst = zipped_divide(gridDomain, gridTiler);
    // Loop through the GEMM problem
    const auto blockCoord = make_coord(blockIdx.x, blockIdx.y);
    // shared memory
    for (unsigned gridIdx = 0; gridIdx < cute::size<1>(grdTiledDst); ++gridIdx) {
        const auto tileIdx = grdTiledDst(blockCoord, gridIdx);
        if (tileIdx >= cute::size(gridDomain)) { continue; }
        auto tileCoord = idx2crd(tileIdx, cute::get<1>(blkTiledDst).shape());
        auto tileCoordA = make_coord(_0{}, cute::get<1>(tileCoord));
        auto tileCoordB = make_coord(cute::get<0>(tileCoord), _0{});
        // Convert 2D tile coordinates to linear indices using crd2idx
        auto tileIdxA = crd2idx(tileCoordA, cute::get<1>(blkTiledA).shape());
        auto tileIdxB = crd2idx(tileCoordB, cute::get<1>(blkTiledB).shape());
        constexpr unsigned int tileK = 32;
        kernel_block<blockTiler, tileK>(
            tileIdxA, tileIdxB, tileIdx,
            blkTiledA, blkTiledB, blkTiledDst, predicate,
            A, B, C, dst);
    }
}

// cudafe++ struggles with CNTTP
template <dim3 grid, dim3 block>
static void kernel_grid_v2_caller(const unsigned m, const unsigned n, const unsigned k,
    const float* A, const float* B, const float* C,
    float* dst) {
    kernel_grid_v2<grid.x, grid.y, grid.z, block.x, block.y, block.z><<<grid, block>>>(m,n,k,A,B,C,dst);
}

void gemm_f32_rrrr_cuda_v2(
    const unsigned m, const unsigned n, const unsigned k,
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
    constexpr dim3 grid = {12,48,1};
    constexpr dim3 block = {32,4,1};
    kernel_grid_v2_caller<grid, block>(m,n,k,matA.ptr(), matB.ptr(), matC.ptr(), matD.ptr());
    cudaMemcpy(dst, matD.ptr(), sizeof(float)*m*n, cudaMemcpyDeviceToHost);
}