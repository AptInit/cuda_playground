enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

add_library(cute_gemm STATIC demo1.cu demo0.cu)
#target_compile_options(cute_gemm PRIVATE --expt-relaxed-constexpr --generate-line-info --extended-lambda)
target_compile_options(cute_gemm PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --generate-line-info
        --extended-lambda
        >
)
#target_compile_options(cute_gemm PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>)
#target_link_options(cute_gemm PRIVATE $<$<LINK_LANGUAGE:CXX>:-fsanitize=address>)
#target_link_libraries(cute_gemm PRIVATE CUDA::cudart)
target_include_directories(cute_gemm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)