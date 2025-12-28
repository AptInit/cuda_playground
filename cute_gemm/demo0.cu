//
// Created by jcw on 12/27/25.
//
#include "cute_gemm.h"
#include "cute/layout.hpp"
#include <ranges>
#include <cassert>

void gemm_f32_row_row_row_row(
    const size_t m, const size_t n, const size_t k,
    const float* A, const float* B, const float* C,
    float* dst) {
    auto get_layout = [](const auto m, const auto n) {
        const auto shape = cute::make_shape(m, n);
        return cute::make_layout(shape, cute::compact_row_major(shape));
    };
    const auto layoutA = get_layout(m, k);
    const auto layoutB = get_layout(k, n);
    const auto layoutC = get_layout(m, n);
    const auto layoutDst = layoutC;
    // The kernel will perform 4xkx4 tiled gemm
    constexpr auto kernelTile = cute::make_shape(cute::_4{}, cute::_4{});
    // create tiler for matrices
    auto tilerA = cute::make_shape(cute::get<0>(kernelTile), k);
    auto tilerB = cute::make_shape(k, cute::get<1>(kernelTile));
    // Tile the problem
    auto tiledA = zipped_divide(layoutA, tilerA);
    auto tiledB = zipped_divide(layoutB, tilerB);
    auto tiledC = zipped_divide(layoutC, kernelTile);
    auto tiledDst = zipped_divide(layoutDst, kernelTile);
    // The kernel needs two levels of info per matrix:
    //  1. Layout(just the size) and ptr of top-level matrix
    //  2. Offset of tile matrix, and layout of the tile
    auto kernel = [&A, &B, &C, &dst,
        &tiledA, &tiledB, &tiledC, &tiledDst,
        kernelTile] <typename PredFunc> (
            const auto tileA_idx, const auto tileB_idx, const auto tile_idx, PredFunc predicate) {
        // Kernel hyperparameters controlled by kernelTile
        constexpr int tileM = cute::get<0>(kernelTile);
        constexpr int tileN = cute::get<1>(kernelTile);
        {
            auto layoutTileA = cute::get<0>(cute::slice(cute::make_coord(cute::_, tileA_idx), tiledA));
            auto layoutTileB = cute::get<0>(cute::slice(cute::make_coord(cute::_, tileB_idx), tiledB));
            // Sanity check
            assert(cute::get<1>(layoutTileA.shape()) == cute::get<0>(layoutTileB.shape()));
        }
        // 1. Initialize Accumulators (Registers)
        float acc[tileM][tileN];

        for (int i = 0; i < tileM; ++i) {
            for (int j = 0; j < tileN; ++j) {
                auto localCoord = cute::make_coord(i, j);
                if (predicate(cute::crd2idx(localCoord, kernelTile), tile_idx)) {
                    auto idx = tiledC(localCoord, tile_idx);
                    acc[i][j] = C[idx];
                } else {
                    acc[i][j] = 0.0f; // Padding handling
                }
            }
        }

        // 2. Main Computation Loop (K-dimension)
        // We iterate over K, updating the accumulators
        const auto dimK = cute::size<1>(cute::shape<0>(tiledA)); // K dimension size

        for (int k_idx = 0; k_idx < dimK; ++k_idx) {
           for (int i = 0; i < tileM; ++i) {
                for (int j = 0; j < tileN; ++j) {
                    auto localCoord = cute::make_coord(i, j);
                    if (predicate(cute::crd2idx(localCoord, kernelTile), tile_idx)) {
                        // Safe to read A (row i) and B (col j)
                        auto localCoordA = cute::make_coord(i, k_idx);
                        auto idxA = tiledA(localCoordA, tileA_idx);

                        auto localCoordB = cute::make_coord(k_idx, j);
                        auto idxB = tiledB(localCoordB, tileB_idx);

                        acc[i][j] += A[idxA] * B[idxB];
                    }
                }
           }
        }

        // 3. Epilogue: Store results
        for (int i = 0; i < tileM; ++i) {
            for (int j = 0; j < tileN; ++j) {
                auto localCoord = cute::make_coord(i, j);
                if (predicate(cute::crd2idx(localCoord, kernelTile), tile_idx)) {
                     auto idxDst = tiledDst(localCoord, tile_idx);
                     dst[idxDst] = acc[i][j];
                }
            }
        }
    };
    // Loop over tiles
    auto tile_cnt = std::ranges::iota_view{size_t{0}, cute::size<1>(tiledDst)};
    std::ranges::for_each(tile_cnt,
        [&] (auto tile_idx) {
            auto tileCoord = cute::idx2crd(tile_idx, cute::get<1>(tiledDst).shape());
            auto tileCoordA = cute::make_coord(cute::get<0>(tileCoord), cute::_0{});
            auto tileCoordB = cute::make_coord(cute::_0{}, cute::get<1>(tileCoord));
            // Get bounds checker layout for each basis
            auto make_basis = [] (auto layout, auto tiler) {
                // Do stuff recursively in the future
                static_assert(cute::rank(layout) == 2);
                return std::make_tuple(
                    cute::zipped_divide(
                        cute::make_layout(layout.shape(), cute::make_stride(cute::_1{}, cute::_0{})), tiler),
                    cute::zipped_divide(
                        cute::make_layout(layout.shape(), cute::make_stride(cute::_0{}, cute::_1{})), tiler));
            };
            auto [boundsDstM, boundsDstN] = make_basis(layoutDst, kernelTile);
            // Convert 2D tile coordinates to linear indices using crd2idx
            auto tileIdxA = cute::crd2idx(tileCoordA, cute::get<1>(tiledA).shape());
            auto tileIdxB = cute::crd2idx(tileCoordB, cute::get<1>(tiledB).shape());
            auto predicate = [&](auto local_idx, auto tile_idx){
                return boundsDstM(local_idx, tile_idx) < cute::size<0>(layoutDst)
                    && boundsDstN(local_idx, tile_idx) < cute::size<1>(layoutDst);};
            // Note that kernel only uses linear indices
            kernel(tileIdxA, tileIdxB, tile_idx, predicate);});
}
