enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

add_library(cute_gemm STATIC demo3.cu demo2.cu demo1.cu demo0.cu)
target_compile_options(cute_gemm PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --generate-line-info
        --extended-lambda
        >
)
if(MSVC)
    target_compile_options(cute_gemm PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus /permissive->
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus -Xcompiler=/permissive->
    )
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(cute_gemm PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
            --keep
            >
    )
endif()
# Auto-tuning support: pass individual parameters to build the BlkCfg type
# Example: cmake -DBENCH_GRD=4 -DBENCH_BLKN=8 -DBENCH_BLKM=8 -DBENCH_BLKK=16 -DBENCH_THN=2 -DBENCH_THM=2
if(DEFINED BENCH_GRD AND DEFINED BENCH_BLKN AND DEFINED BENCH_BLKM AND DEFINED BENCH_BLKK AND DEFINED BENCH_THN AND DEFINED BENCH_THM)
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/demo3.cu
        PROPERTIES COMPILE_DEFINITIONS 
            "BENCH_GRD=${BENCH_GRD};BENCH_BLKN=${BENCH_BLKN};BENCH_BLKM=${BENCH_BLKM};BENCH_BLKK=${BENCH_BLKK};BENCH_THN=${BENCH_THN};BENCH_THM=${BENCH_THM}"
    )
endif()
#target_compile_options(cute_gemm PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>)
#target_link_options(cute_gemm PRIVATE $<$<LINK_LANGUAGE:CXX>:-fsanitize=address>)
#target_link_libraries(cute_gemm PRIVATE CUDA::cudart)
target_include_directories(cute_gemm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)