# A pre-tensor core CUDA GEMM Walkthrough: `kernel_grid_v3`

While previous demos focused on the basics of **Layout Algebra**, this kernel is a "not-so-toyish" implementation that incorporates common GPU micro-architecture optimizations: **shared memory tiling**, **thread tiling**, and **bank conflict avoidance via swizzled layouts**.

## 1. The Configuration: `BlkCfg`

High-performance CUDA kernels are often parameterized by tile sizes that must be tuned for specific hardware.

```cpp
template <unsigned grd_, unsigned blkN_, unsigned blkM_, unsigned blkK_,
    unsigned thN_, unsigned thM_>
struct BlkCfg {
  using BlkN = cute::Int<blkN_>; // SMEM Tile N
  using BlkM = cute::Int<blkM_>; // SMEM Tile M
  using BlkK = cute::Int<blkK_>; // SMEM Tile K
  using ThN = cute::Int<thN_>;   // Register Tile N (per thread)
  using ThM = cute::Int<thM_>;   // Register Tile M (per thread)
  // ...
};
```

### Why Compile-Time Constants?
In Modern C++, we use `cute::Int<N>` (a singleton type) to pass constants through the type system. This allows the compiler to:
1.  **Statically Allocate SMEM**: The size is known at compile time.
2.  **Aggressively Unroll Loops**: The trip counts are constants.
3.  **Optimize Register Usage**: The accumulator size is fixed, letting the compiler map it directly to hardware registers ($R0, R1, \dots$).

---

## 2. Memory Hierarchy and Tiling Strategy

The kernel follows the standard GPU GEMM hierarchy: **Global Memory $\to$ Shared Memory $\to$ Registers**.

### 2.1 Grid-Level Tiling (`kernel_grid_v3`)
The first step is mapping our 2D grid of Thread Blocks (CTAs) to the 2D output matrix $C$.

```cpp
const auto gridTiler = make_tile(Int<grdDim>{}, k, Int<grdDim>{});
const auto blockTiler = make_tile(typename Cfg::BlkN{} * typename Cfg::ThN{},
      k, typename Cfg::BlkM{} * typename Cfg::ThM{});
```

- **Calculated Block Size**: Note that the actual tile size produced by a block is `(BlkN * ThN, BlkM * ThM)`. 
- **CTA Predication**: The code uses `cTileCoord` and `elem_less` to check if a Block is entirely outside the matrix boundaries. If it's on the edge, it switches to a "safe" version of the kernel (`Cfg{false}`) that performs element-wise predication.

### 2.2 Global Layouts
CuTe uses **colexicographical** ordering by default (fastest varying dimension first). Since our input matrices $A, B, C$ are row-major in memory ($m \times k$ means $m$ is the outer dimension), we define them accordingly:

```cpp
const auto aTen = make_tensor(A, make_layout(make_shape(k, m))); // k is inner
```

---

## 3. The Block Level: `kernel_block`

Once inside a block, we allocate Shared Memory (SMEM) to store tiles of $A$ and $B$. This reduces the pressure on Global Memory bandwidth.

### 3.1 Defining Shared Memory Layouts
In `kernel_block`, we define the layouts for our Shared Memory tiles. We use a **Swizzled Layout** for $A$ to facilitate high-performance loads later in the compute loop.

```cpp
// Layout for A in SMEM (Swizzled)
constexpr auto aSMEMSwizzle = 
    cute::Swizzle<2, log_2(static_cast<unsigned>(cfg.thM)), 4>{};
constexpr auto aSMEMLayoutBase = 
    make_layout(make_shape(cfg.blkK, cfg.blkM * cfg.thM));
constexpr auto aSMEMLayout = cute::composition(aSMEMSwizzle, aSMEMLayoutBase);

// Layout for B in SMEM (Linear)
constexpr auto bSMEMLayout =
    make_layout(make_shape(cfg.blkN * cfg.thN, cfg.blkK));
```

- **Permutation via Composition**: The `cute::composition` of a `Swizzle` and a `Layout` essentially "shuffles" the logical mapping of coordinates to physical offsets. We'll explore the micro-architectural necessity of this in Section 4.2.

---

## 4. The Main Loop: `tiny_loop`

The `tiny_loop` is where the heavy lifting happens. It is divided into two phases: **Copy** and **Compute**.

### 4.1 Copy (Global $\to$ SMEM)
We use the thread's `threadIdx` to cooperatively load a tile from Global Memory into SMEM.

```cpp
for (unsigned laOffset = threadIdx.x; laOffset < cfg.blkK; laOffset += cfg.blkN) {
    aSTen(laOffset, threadIdx.y + tm * cfg.blkM) = a_ttiled(...);
}
```

### 4.2 Compute (SMEM $\to$ Registers)
Each thread maintains a private accumulator tile (`accTile`) of size `ThN x ThM`. The compute logic needs a rather sophisticated vectorization strategy:

1.  **A-Vector Load**: It loads a vector of `thM` elements from $A$ along the **K-dimension**. This is possible because our SMEM layout is column-major ($K$ is the first, stride-1 dimension).
2.  **B-Vector Load**: For each $k \in [kBase, kBase+thM)$, it loads a vector of `thN` elements from $B$ along the **N-dimension**. This is also stride-1.
3.  **Outer-Product**: It then performs the `accTile(tn, tm) += aTmp(kI) * bTmp(tn)` updates.

```cpp
for (unsigned kBase = 0; kBase < cfg.blkK; kBase += cfg.thM) {
    for (unsigned tm = 0; tm < cfg.thM; ++tm) {
        // Load A vector along K (stride-1)
        util::load_vectorized<cfg.thM, float>(...); 
        
        for (unsigned kI = 0; kI < cfg.thM; ++kI) {
            // Load B vector along N (stride-1)
            util::load_vectorized<cfg.thN, float>(...);
            
            for (unsigned tn = 0; tn < cfg.thN; ++tn) {
                accTile(tn, tm) += aTmp(kI) * bTmp(tn);
            }
        }
    }
}
```

This strategy ensures that every single SMEM access is a **vectorized load** (`LDS.64` or higher depending on `ThM` and `ThN`), minimizing shared memory accesses.

#### Bank Conflict Avoidance
This is where the **Swizzled Layout** for `aSTen` defined in section 3.1 becomes critical.

- **The Problem**: Shared Memory is organized into 32 banks (4 bytes each). If multiple threads in a warp access different addresses that map to the same bank, we get a **bank conflict**, which serializes the access threads and throttles throughput.
- **Stride-Induced Conflicts**: In the compute loop, threads often access SMEM with large strides (e.g., when reading tiles for thread-tiling). If this stride is a multiple of 32, every thread in a warp might hit the exact same bank.
- **The Solution**: `cute::Swizzle` applying an XOR-based permutation to the logical address bits. This "shuffles" the physical bank mapping so that even with large logical strides, the physical bank indices are distributed across the warp.

---

## 5. Layout Algebra: The "Brain"

Notice that nowhere in the kernel do we see manual pointer arithmetic like `A[blockIdx.x * stride + threadIdx.x]`. Instead, we use `local_tile`, `zipped_divide`, and `composition`.

- **Abstraction**: `local_tile(aTen, blockTiler, tileCoord)` slices the global matrix down to exactly what the CTA needs to see.
- **Independence**: This separation of **Logic** (what the data is) from **Algebra** (how we view the data) is why CuTe is transformative. You can change a matrix from Row-Major to Column-Major by altering a single `Layout` line, and the rest of the kernel—including the swizzled SMEM logic—remains mathematically correct.

---

## 6. Micro-Architecture & Performance Insights

### 6.1 The MIO Bottleneck
If you profile this kernel in **Nsight Compute**, the top stall reason is likely **MIO Throttled** (Memory Input/Output). This means the Shared Memory (SMEM) units are saturated.

- **The Naive Thought**: "Can I read 32 values from `aSTen` into registers once, then use warp-shuffles (`__shfl_sync`) to pass them around and save SMEM bandwidth?"
- **The Reality**: Warp-level intrinsics use the same underlying crossbar/routing hardware as Shared Memory. Bypassing SMEM with shuffles doesn't actually bypass the bottleneck—you're just hitting the same limit from a different instruction.

### 6.2 SM Sub-Partitions and Ports
Modern NVIDIA SMs (since Volta/Ampere) are divided into **four sub-partitions** (SMSPs). Each sub-partition has its own scheduler and register file, but they **share the same port** to access the Shared Memory unit. 

This shared resource contention is why we must be obsessed with hierarchies. The movement from SMEM $\to$ Registers is a precious, shared resource that must be managed with precise timing and swizzled layouts to prevent the SMSPs from stalling each other.

### 6.3 Integer vs. Floating Point Pipeline
One subtle cost of high-level abstractions like CuTe is **Integer Arithmetic** overhead for coordinate calculations.
- Currently, this kernel performs a significant amount of INT ops to resolve layouts.
- While CuTe's `Int<N>` types move many calculations to compile-time, the resulting kernel still spends considerable "INT pipeline" time. Optimizing these arithmetic operations is a top item on the To-Do list.

### 6.4 Thread-Tiling: The Path to Tensor Cores
We use **Thread-Tiling** (`ThN`, `ThM`) to reduce MIO pressure. By increasing these values, more math per SMEM load can be performed. 
- As these dimensions grow, the thread starts to behave like a **systolic array**, where a slice of column in `A` and a slice of row in `B` are feed into the FPUs in each step.
- However, there is a hard limit: **Register Pressure**. If `ThN * ThM` is too large, a single thread will exhaust its quota (255 registers), causing "register spilling" to slow local memory. Besides, micro-architecture is optimized for up-to 16 bytes per thread, thus the maximum `ThN * ThM` is 16.

**All these are reasons for Tensor Cores.** Tensor Cores are hardware-accelerated matrix-multiply units. They perform a large outer product (e.g., $16 \times 8 \times 16$) in a single step, bypassing the register-pressure (and later MIO-bandwidth) constraints of general-purpose CUDA cores by performing the math inside a dedicated, high-density silicon block.

### 6.5 GMEM Alignment and Vectorization
A factor for improving Global Memory (GMEM) access is **vectorization**. To achieve peak bandwidth, the hardware should ideally issue 128-bit loads (`LDG.128`).

- **The Alignment Requirement**: For these 128-bit wide instructions to work, the memory address must be 16-byte aligned. 
- **The Problem of Unaligned Matrices**: In practice, matrices are not always perfectly aligned to 16-byte boundaries. 
To handle this, kernel should split the memory copy into a **Prologue-Mainloop-Epilogue** structure:
    1.  **Prologue**: Handle the initial few elements using smaller (1, 2, 4, or 8-byte) loads until a 16-byte boundary is reached.
    2.  **Main Loop**: Execute the bulk of the data transfer using the most efficient vectorized 128-bit loads.
    3.  **Epilogue**: Clean up any remaining "trailing" elements that don't fill a full 16-byte vector.

This kernel didn't implement this optimization, partly because GMEM access is not a major bottleneck, which can be shown by reverting the grid-level tiling and profiling the (lack of) difference.

---

## 7. Summary of Optimization Path

| Feature | Performance Benefit | GPU Micro-Arch Insight |
| :--- | :--- | :--- |
| **SMEM Tiling** | Reuses data, reduces Global Memory traffic. | Hides Global L2 latency. |
| **Thread Tiling** | Increases arithmetic intensity. | Reduces MIO (SMEM $\to$ Reg) pressure. |
| **Swizzling** | Eliminates SMEM Bank Conflicts. | Saturation of the shared SMEM port. |
| **Static Cfg** | Allows compiler to unroll and optimize. | Minimizes INT pipeline overhead for layouts. |

This `v3` kernel represents a reasonable baseline for GEMM on modern GPUs, balancing the complex trade-offs between different hardware pipelines.

---

## 8. References

For further reading on optimizing CUDA GEMM kernels and understanding the underlying principles, see:

1. [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) - A deep dive into the optimization journey of a CUDA matrix multiplication kernel.
2. [Learn CUTLASS the Hard Way](https://www.kapilsharma.dev/posts/learn-cutlass-the-hard-way/) - A comprehensive guide to understanding the internals and design philosophy of the NVIDIA CUTLASS library.
