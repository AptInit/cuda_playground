cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_ARCHITECTURES "120;86")
project(cuda_playground LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
#add_subdirectory(cutlass)
add_subdirectory(cute_gemm)
add_executable(cuda_playground main.cpp)
target_link_libraries(cuda_playground PRIVATE cute_gemm)

#include(GNUInstallDirs)
#install(TARGETS cuda_playground
#        RUNTIME DESTINATION bin  # Will be <prefix>/bin
#        LIBRARY DESTINATION lib  # Will be <prefix>/lib
#)