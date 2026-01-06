# Walkthrough of the toy GEMM kernel

This document provides a walkthrough of a CPU-based blocked Matrix Multiplication (GEMM) implementation using **CuTe**, the layout algebra engine powering NVIDIA's CUTLASS library.

> **Why CPU?** 
> Writing GPU kernels involves managing threads, shared memory, and warp synchronization. By running this on the CPU, we strip away those hardware complexities and focus entirely on **Layout Algebra**: how to decompose matrices into tiles and map coordinates to memory indices using CuTe.

## 1. Defining the Problem

We are implementing a simple Matrix Multiplication: $C = A \times B$.
- **Layouts**: All matrices ($A, B, C$) are Row-Major.
- **Tiling**: We will compute the result in **4x4 tiles**.
- **Strategy**: We will loop over the output matrix $C$ one 4x4 tile at a time. For each tile, we compute the dot product of the corresponding strip of $A$ and strip of $B$.

## 2. Layouts and Tiling

In CuTe, everything starts with a **Layout**, which is a mapping from logical coordinates to physical integers (memory offsets).

```cpp
auto get_layout = [](const auto m, const auto n) {
    const auto shape = cute::make_shape(m, n);
    return cute::make_layout(shape, cute::compact_row_major(shape));
};
const auto layoutA = get_layout(m, k);
```

### The Magic of `zipped_divide`
To implement a tiled algorithm, we need to view our matrix not as a flat grid of NxM, but as a "grid of tiles". CuTe achieves this with `zipped_divide()`.

```cpp
constexpr auto kernelTile = cute::make_shape(cute::_4{}, cute::_4{});
auto tiledC = zipped_divide(layoutC, kernelTile);
```

`zipped_divide` splits the original layout into two levels:
1.  **The Atom**: The tile itself (4x4).
2.  **The Rest**: The grid of tiles.

If `layoutC` was 16x16, `tiledC` acts like a 4D tensor of shape `((4, 4), (4, 4))`. 
- The inner components `(4, 4)` are the **Tile**.
- The outer components `(4, 4)` are the **Rest** (coordinates of the tile in the grid).

## 3. The Outer Loop (Visiting Tiles)

We iterate over the "Rest" dimension—essentially stepping through every 4x4 block of $C$.

```cpp
// Iterate 0, 1, 2... N_Tiles
auto tile_cnt = std::ranges::iota_view{size_t{0}, cute::size<1>(tiledDst)};

std::ranges::for_each(tile_cnt, [&] (auto tile_idx) {
    // ...
});
```

Inside this loop, we have a single linear index `tile_idx`. We need to know which (Row, Col) tile this corresponds to.

```cpp
// Convert linear tile index -> (BlockRow, BlockCol)
auto tileCoord = cute::idx2crd(tile_idx, cute::get<1>(tiledDst).shape());
```

From this `tileCoord` (a tuple like `{2, 3}`), we figure out which row of tiles in $A$ and which column of tiles in $B$ we need.

## 4. Bounds Checking with "Basis" Layouts

One of the hardest parts of tiling is handling edges (e.g., if matrix size is 10 but tile size is 4, the last tile is partial).

Layout algebra provides a solution: **Basis Layouts**.
It creates dummy layouts that don't point to data, but rather return the *logical coordinate* itself (Row index or Column index).

```cpp
auto [boundsDstM, boundsDstN] = make_basis(layoutDst, kernelTile);
```
- `boundsDstM(local_idx, tile_idx)` returns the global Row index for a specific element.
- `boundsDstN(local_idx, tile_idx)` returns the global Col index for a specific element.

We use these to build a **Predicate**:
```cpp
auto predicate = [&](auto local_idx, auto tile_idx){
    return boundsDstM(local_idx, tile_idx) < M && 
           boundsDstN(local_idx, tile_idx) < N;
};
```
This is passed into the kernel to safely mask off out-of-bounds reads/writes.

## 5. The "Kernel" (Inner Logic)

This lambda function mimics what a GPU CUDA kernel would do. It operates on a **single tile**.

### Usage of Linear Indices
Notice the kernel signature takes *linear indices* for the tiles, adhering to the CuTe idiom that "Indices are simple, Coordinates are rich".

```cpp
kernel(tileIdxA, tileIdxB, tile_idx, predicate);
```

### Systolic-Array-Style Accumulation
The inner logic mimics a hardware systolic array or a register-tiled GPU kernel.

1.  **Initialize Registers**: We create a local `acc[4][4]` array (simulating GPU registers). We load the initial value of $C$ (or 0 if out of bounds).
    
    ```cpp
    float acc[tileM][tileN]; 
    // Load C...
    ```

2.  **The K-Loop**: We loop over the K-dimension.
    Notice how we access memory. We don't do pointer arithmetic manually (`A + i*stride + ...`). We just ask the layout!
    
    ```cpp
    auto idxA = tiledA(cute::make_coord(i, k_idx), tileA_idx);
    ```
    
    `tiledA` knows how to map the *local* coordinate `(i, k)` within the tile `tileA_idx` to the correct global index in the `A` array.

3.  **Epilogue**: Finally, we write the accumulator back to `dst`, respecting the predicate to avoid writing out of bounds.

## Summary

This "toy" kernel demonstrates the power of CuTe:
- **Abstraction**: We reasoned about Tiles and Coordinates, not Pointers and Offsets.
- **Safety**: Bounds checking comes naturally with the layout.
- **Correctness**: The `zipped_divide` ensured we mathematically partitioned the matrix correctly.

However, CuTe also provides some utilities to make this even easier, including:
 - cute::Tensor()
 - cute::make_identity_tensor()

Moving this to a GPU basically involves changing `for` loops to `threadIdx` mapping and `acc` arrays to actual registers—the Layout Algebra remains exactly the same.
