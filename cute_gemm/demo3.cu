#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute_gemm.h"
#include "util.hpp"

template <unsigned grd_, unsigned blkN_, unsigned blkM_, unsigned blkK_,
    unsigned thN_, unsigned thM_>
struct BlkCfg {
  // Skip predication on N and M?
  bool nMPredSkip{};
  // blockDim.x
  using BlkN = cute::Int<blkN_>;
  BlkN blkN{};
  // blockDim.y
  using BlkM = cute::Int<blkM_>;
  BlkM blkM{};
  using BlkK = cute::Int<blkK_>;
  BlkK blkK{};
  // Each thread covers a (thN, thM) tile
  using ThN = cute::Int<thN_>;
  ThN thN{};
  using ThM = cute::Int<thM_>;
  ThM thM{};
  // Subgrid size for L2 cache reuse
  using Grd = cute::Int<grd_>;
  Grd grd{};
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
        bSTen(threadIdx.x + tn * cfg.blkN, lbOffset + threadIdx.y) =
            b_ttiled(threadIdx.x + tn * cfg.blkN, lbOffset + threadIdx.y);
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
  for (unsigned kBase = 0; kBase < cfg.blkK; kBase += cfg.thM) {
    for (unsigned tm = 0; tm < cfg.thM; ++tm) {
      auto aTmp = make_tensor<float>(make_shape(cfg.thM));
      util::load_vectorized<cfg.thM, float>(
          &aTmp(_0{}), &aSTen(kBase, threadIdx.y * cfg.thM + tm));
      for (unsigned kI = 0; kI < cfg.thM; ++kI) {
        const auto k = kBase + kI;
        auto bTmp = make_tensor<float>(make_shape(cfg.thN));
        util::load_vectorized<cfg.thN, float>(
            &bTmp(_0{}), &bSTen(threadIdx.x * cfg.thN, k));
        for (unsigned tn = 0; tn < cfg.thN; ++tn) {
          accTile(tn, tm) += aTmp(kI) * bTmp(tn);
        }
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
  constexpr auto aSMEMSwizzle =
      cute::Swizzle<2, log_2(static_cast<unsigned>(cfg.thM)), 4>{};
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

template <unsigned grd_, unsigned blkN_, unsigned blkM_, unsigned blkK_,
    unsigned thN_, unsigned thM_>
__global__ static void __launch_bounds__(blkN_* blkM_)
    kernel_grid_v3(const unsigned m, const unsigned n, const unsigned k,
        const float* A, const float* B, const float* C, float* dst) {
  // Integer intensity is ridiculous
  using namespace cute;
  namespace cg = cooperative_groups;


  // Build tilers based on kernel config
  //  Tunable
  using Cfg = BlkCfg<grd_, blkN_, blkM_, blkK_, thN_, thM_>;
  constexpr unsigned grdDim =
      util::static_powk(util::static_logk(grd_, 4u), 2u);
  // Build tilers
  const auto gridTiler = make_tile(Int<grdDim>{}, k, Int<grdDim>{});
  const auto blockTiler = make_tile(typename Cfg::BlkN{} * typename Cfg::ThN{},
      k, typename Cfg::BlkM{} * typename Cfg::ThM{});


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
  const auto cTileCoord = cGrdPred(
      make_coord(blockIdx.x % grdDim, blockIdx.x / grdDim), blockIdx.y);
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
    if (elem_less(probITiled(typename Cfg::BlkN{} * typename Cfg::ThN{} - _1{},
                      _0{}, typename Cfg::BlkM{} * typename Cfg::ThM{} - _1{}),
            world)) {
      kernel_block<Cfg{true}>(
          aTiled, bTiled, cTiled, dstTiled, probITiled, shape(world));
    } else {
      kernel_block<Cfg{false}>(
          aTiled, bTiled, cTiled, dstTiled, probITiled, shape(world));
    }
  }
}

// cudafe++ struggles with CNTTP
template <BlkCfg cfg>
static void kernel_grid_v3_caller(const unsigned m, const unsigned n,
    const unsigned k, const float* A, const float* B, const float* C,
    float* dst) {
  // Kernel caller's responsibility to set the correct grid dimensions
  dim3 grid{};
  grid.x = cfg.grd;
  constexpr unsigned grdDim = util::static_powk(
      util::static_logk(static_cast<unsigned>(cfg.grd), 4u), 2u);
  grid.y = cute::ceil_div(n, grdDim * cfg.blkN * cfg.thN) *
           cute::ceil_div(m, grdDim * cfg.blkM * cfg.thM);
  grid.z = 1;
  dim3 block = {
      static_cast<unsigned>(cfg.blkN), static_cast<unsigned>(cfg.blkM), 1};
  kernel_grid_v3<cfg.grd, cfg.blkN, cfg.blkM, cfg.blkK, cfg.thN, cfg.thM>
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
  using Cfg = BlkCfg<16, 32, 32, 32, 2, 2>;
  kernel_grid_v3_caller<Cfg{}>(
      m, n, k, matA.ptr(), matB.ptr(), matC.ptr(), matD.ptr());
  cudaMemcpy(dst, matD.ptr(), sizeof(float) * m * n, cudaMemcpyDeviceToHost);
}