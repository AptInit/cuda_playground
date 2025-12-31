enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

add_library(cute_gemm STATIC demo2.cu demo1.cu demo0.cu)
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
#target_compile_options(cute_gemm PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>)
#target_link_options(cute_gemm PRIVATE $<$<LINK_LANGUAGE:CXX>:-fsanitize=address>)
#target_link_libraries(cute_gemm PRIVATE CUDA::cudart)
target_include_directories(cute_gemm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)