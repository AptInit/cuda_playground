#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute_gemm.h"
#include "util.hpp"

template <unsigned blkN_, unsigned blkM_, unsigned blkK_, unsigned thN_,
    unsigned thM_>
struct BlkCfg {
  // Skip predication on N and M?
  bool nMPredSkip{};
  // blockDim.x
  cute::Int<blkN_> blkN{};
  // blockDim.y
  cute::Int<blkM_> blkM{};
  cute::Int<blkK_> blkK{};
  // Each thread covers a (thN, thM) tile
  using ThN = cute::Int<thN_>;
  ThN thN{};
  using ThM = cute::Int<thM_>;
  ThM thM{};
};

template <BlkCfg cfg, bool kPredSkip, typename TACCR, typename TAS,
    typename TBS, typename TA, typename TB, typename TPred, typename TWldShape>
__device__ __forceinline__ static void tiny_loop(TACCR& accTile, TAS aSTen,
    TBS bSTen, const TA a_ttiled, const TB b_ttiled, const TPred id_ttiled,
    const TWldShape worldShape) {
  using namespace cute;
  // load scratch
  static_assert(cfg.blkK % cfg.blkN == 0);
  __syncthreads();
  if constexpr (cfg.nMPredSkip && kPredSkip) {
    for (unsigned tm = 0; tm < cfg.thM; ++tm) {
      for (unsigned laOffset = threadIdx.x; laOffset < cfg.blkK;
          laOffset += cfg.blkN) {
        aSTen(laOffset, threadIdx.y + tm * cfg.blkM) =
            a_ttiled(laOffset, threadIdx.y + tm * cfg.blkM);
      }
    }

    for (unsigned lbOffset = 0; lbOffset < cfg.blkK; lbOffset += cfg.blkM) {
      for (unsigned tn = 0; tn < cfg.thN; ++tn) {
        bSTen(threadIdx.x * cfg.thN + tn, lbOffset + threadIdx.y) =
            b_ttiled(threadIdx.x * cfg.thN + tn, lbOffset + threadIdx.y);
      }
    }
  } else {
    for (unsigned tm = 0; tm < cfg.thM; ++tm) {
      for (unsigned laOffset = threadIdx.x; laOffset < cfg.blkK;
          laOffset += cfg.blkN) {
        const bool loadA =
            (cfg.nMPredSkip ||
                elem_less(id_ttiled(_0{}, _0{}, threadIdx.y + tm * cfg.blkM),
                    worldShape)) &&
            (kPredSkip ||
                elem_less(id_ttiled(_0{}, laOffset, _0{}), worldShape));
        aSTen(laOffset, threadIdx.y + tm * cfg.blkM) =
            loadA ? a_ttiled(laOffset, threadIdx.y + tm * cfg.blkM) : 0.0f;
      }
    }
    for (unsigned lbOffset = threadIdx.y; lbOffset < cfg.blkK;
        lbOffset += cfg.blkM) {
      for (unsigned tn = 0; tn < cfg.thN; ++tn) {
        const bool loadB =
            (cfg.nMPredSkip ||
                elem_less(id_ttiled(threadIdx.x * cfg.thN + tn, _0{}, _0{}),
                    worldShape)) &&
            (kPredSkip ||
                elem_less(id_ttiled(_0{}, lbOffset, _0{}), worldShape));
        bSTen(threadIdx.x * cfg.thN + tn, lbOffset) =
            loadB ? b_ttiled(threadIdx.x * cfg.thN + tn, lbOffset) : 0.0f;
      }
    }
  }
  // compute
  __syncthreads();
  for (unsigned k = 0; k < cfg.blkK; ++k) {
    for (unsigned tm = 0; tm < cfg.thM; ++tm) {
      for (unsigned tn = 0; tn < cfg.thN; ++tn) {
        accTile(tn, tm) += aSTen(k, threadIdx.y * cfg.thM + tm) *
                           bSTen(threadIdx.x * cfg.thN + tn, k);
      }
    }
  }
}

template <BlkCfg cfg, typename TA, typename TB, typename TC, typename TDst,
    typename TPred, typename TWldShape>
__device__ __forceinline__ static void kernel_block(const TA a_tiled,
    const TB b_tiled, const TC c_tiled, const TDst dst_tiled,
    const TPred id_tiled, const TWldShape worldShape) {
  using namespace cute;
  // Swizzled layout for aSTen to reduce SMEM bank conflicts
  constexpr auto aSMEMSwizzle = cute::Swizzle<2, 0, 4>{};
  constexpr auto aSMEMLayoutBase =
      make_layout(make_shape(cfg.blkK, cfg.blkM * cfg.thM));
  constexpr auto aSMEMLayout = cute::composition(aSMEMSwizzle, aSMEMLayoutBase);
  __shared__ float aS[decltype(cosize(aSMEMLayoutBase))::value];
  auto aSTen = make_tensor(make_smem_ptr(aS), aSMEMLayout);
  constexpr auto bSMEMLayout =
      make_layout(make_shape(cfg.blkN * cfg.thN, cfg.blkK));
  __shared__ float bS[decltype(cosize(bSMEMLayout))::value];
  auto bSTen = make_tensor(make_smem_ptr(bS), bSMEMLayout);
  auto accTile = make_tensor<float>(make_shape(cfg.blkN, cfg.blkM));
  if constexpr (cfg.nMPredSkip) {
    for (unsigned tm = 0; tm < cfg.thM; ++tm) {
      for (unsigned tn = 0; tn < cfg.thN; ++tn) {
        accTile(tn, tm) =
            c_tiled(threadIdx.x * cfg.thN + tn, threadIdx.y * cfg.thM + tm);
      }
    }
  } else {
    for (unsigned tm = 0; tm < cfg.thM; ++tm) {
      for (unsigned tn = 0; tn < cfg.thN; ++tn) {
        const bool isValid = elem_less(id_tiled(threadIdx.x * cfg.thN + tn,
                                           _0{}, threadIdx.y * cfg.thM + tm),
            worldShape);
        accTile(tn, tm) = isValid ? c_tiled(threadIdx.x * cfg.thN + tn,
                                        threadIdx.y * cfg.thM + tm)
                                  : 0.0f;
      }
    }
  }

  // Tile over K, this should be done at upper level TBH
  //  Compiler should optimize this through inlining
  const bool partialK = get<1>(worldShape) % cfg.blkK;
  const unsigned kCnt = get<1>(worldShape) / cfg.blkK;
  for (unsigned tileIdx = 0; tileIdx < kCnt; ++tileIdx) {
    tiny_loop<cfg, true>(accTile, aSTen, bSTen,
        local_tile(a_tiled, aSMEMLayout.shape(), tileIdx),
        local_tile(b_tiled, bSMEMLayout.shape(), tileIdx),
        local_tile(id_tiled,
            make_tile(cfg.blkN * cfg.thN, cfg.blkK, cfg.blkM * cfg.thM),
            tileIdx),
        worldShape);
  }
  if (partialK) {
    tiny_loop<cfg, false>(accTile, aSTen, bSTen,
        local_tile(a_tiled, aSMEMLayout.shape(), kCnt),
        local_tile(b_tiled, bSMEMLayout.shape(), kCnt),
        local_tile(id_tiled,
            make_tile(cfg.blkN * cfg.thN, cfg.blkK, cfg.blkM * cfg.thM), kCnt),
        worldShape);
  }
  if constexpr (cfg.nMPredSkip) {
    for (unsigned tm = 0; tm < cfg.thM; ++tm) {
      for (unsigned tn = 0; tn < cfg.thN; ++tn) {
        dst_tiled(threadIdx.x * cfg.thN + tn, threadIdx.y * cfg.thM + tm) =
            accTile(tn, tm);
      }
    }
  } else {
    for (unsigned tm = 0; tm < cfg.thM; ++tm) {
      for (unsigned tn = 0; tn < cfg.thN; ++tn) {
        const bool isValid = elem_less(id_tiled(threadIdx.x * cfg.thN + tn,
                                           _0{}, threadIdx.y * cfg.thM + tm),
            worldShape);
        if (isValid)
          dst_tiled(threadIdx.x * cfg.thN + tn, threadIdx.y * cfg.thM + tm) =
              accTile(tn, tm);
      }
    }
  }
}

template <unsigned grid_x, unsigned grid_y, unsigned grid_z, unsigned block_x,
    unsigned block_y, unsigned block_z>
__global__ static void __launch_bounds__(block_x* block_y)
    kernel_grid_v3(const unsigned m, const unsigned n, const unsigned k,
        const float* A, const float* B, const float* C, float* dst) {
  // Integer intensity is ridiculous
  using namespace cute;
  namespace cg = cooperative_groups;
  constexpr dim3 Grid_{grid_x, grid_y, grid_z},
      Block_{block_x, block_y, block_z};
  constexpr auto Grid = util::SDim3<Grid_>{};
  constexpr auto Block = util::SDim3<Block_>{};
  static_assert(Grid.z == 1, "2-D grid");


  // Build tilers based on kernel config
  //  Tunable
  using Cfg = BlkCfg<Block.x, Block.y, 32, 2, 2>;
  // Build tilers
  const auto gridTiler = make_tile(Grid.x, k, Grid.y);
  const auto blockTiler = make_tile(
      Block.x * typename Cfg::ThN{}, k, Block.y * typename Cfg::ThM{});


  // Define problem space
  const auto world = make_shape(n, k, m);
  const auto worldI = make_identity_tensor(world);
  // All matrices are row-major, but CuTe uses colexicographical order, follow
  // CuTe's convention
  const auto aTen = make_tensor(A, make_layout(make_shape(k, m)));
  const auto bTen = make_tensor(B, make_layout(make_shape(n, k)));
  const auto cTen = make_tensor(C, make_layout(make_shape(n, m)));
  auto dstTen = make_tensor(dst, cTen.layout());
  // CuTe severely lacks hierarchy manipulation tools
  //  This is a really roundabout way to do CTA-level predication
  // Build hierarchy bottom-up
  const auto cBlkTiled = zipped_divide(cTen.layout(), select<0, 2>(blockTiler));
  const auto cGrdTiled =
      zipped_divide(get<1>(cBlkTiled), select<0, 2>(gridTiler));
  const auto cHier = make_layout(get<0>(cBlkTiled), cGrdTiled);
  // CTA-wide predication
  const auto cGrdPred = zipped_divide(
      make_identity_tensor(shape<1>(cBlkTiled)), select<0, 2>(gridTiler));
  const auto blockCoord = make_coord(blockIdx.x, blockIdx.y);
  for (unsigned gridIdx = 0; gridIdx < size<1>(cGrdTiled); ++gridIdx) {
    const auto cTileCoord = cGrdPred(blockCoord, gridIdx);
    if (elem_less(cTileCoord, shape<1>(cBlkTiled))) {
      // Do stuff
      const auto tileCoord =
          make_coord(get<0>(cTileCoord), _0{}, get<1>(cTileCoord));
      const auto aTiled =
          local_tile(aTen, blockTiler, tileCoord, Step<X, _1, _1>{});
      const auto bTiled =
          local_tile(bTen, blockTiler, tileCoord, Step<_1, _1, X>{});
      const auto cTiled =
          local_tile(cTen, blockTiler, tileCoord, Step<_1, X, _1>{});
      const auto dstTiled =
          local_tile(dstTen, blockTiler, tileCoord, Step<_1, X, _1>{});
      const auto probITiled = local_tile(worldI, blockTiler, tileCoord);
      if (elem_less(probITiled(Block.x * typename Cfg::ThN{} - _1{}, _0{},
                        Block.y * typename Cfg::ThM{} - _1{}),
              world)) {
        kernel_block<Cfg{true}>(
            aTiled, bTiled, cTiled, dstTiled, probITiled, shape(world));
      } else {
        kernel_block<Cfg{false}>(
            aTiled, bTiled, cTiled, dstTiled, probITiled, shape(world));
      }
    }
  }
}

// cudafe++ struggles with CNTTP
template <dim3 grid, dim3 block>
static void kernel_grid_v3_caller(const unsigned m, const unsigned n,
    const unsigned k, const float* A, const float* B, const float* C,
    float* dst) {
  kernel_grid_v3<grid.x, grid.y, grid.z, block.x, block.y, block.z>
      <<<grid, block>>>(m, n, k, A, B, C, dst);
}

void gemm_f32_rrrr_cuda_v3(const unsigned m, const unsigned n, const unsigned k,
    const float* A, const float* B, const float* C, float* dst) {
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
    std::cout << deviceProp.name << std::endl;
  }
  cudaSetDevice(0);
  const auto matA = CuPtr(m * k, A);
  const auto matB = CuPtr(k * n, B);
  const auto matC = CuPtr(m * n, C);
  auto matD = CuPtr<float>(m * n);
  // Kernel
  constexpr dim3 grid = {64, 64, 1};
  constexpr dim3 block = {32, 32, 1};
  kernel_grid_v3_caller<grid, block>(
      m, n, k, matA.ptr(), matB.ptr(), matC.ptr(), matD.ptr());
  cudaMemcpy(dst, matD.ptr(), sizeof(float) * m * n, cudaMemcpyDeviceToHost);
}