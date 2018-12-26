# cmake_minimum_required(VERSION <specify CMake version here>)
project(faiss_v)
#
#option(BUILD_TUTORIAL "Build tutorials" ON)
#option(BUILD_TEST "Build tests" ON)
#option(BUILD_WITH_GPU "Build faiss with gpu (cuda) support" ON)
#option(WITH_MKL "Build with MKL if ON (OpenBLAS if OFF)" OFF)
#
#list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
#
## OpenMP
#find_package(OpenMP REQUIRED)
#
## BLAS (MKL os OpenBLAS)
#if(WITH_MKL)
#    find_package(MKL REQUIRED)
#    include_directories(${MKL_INCLUDE_DIRS})
#    set(BLAS_LIB ${MKL_LIBRARIES})
#else()
#    find_package(OpenBLAS REQUIRED)
#    include_directories(${OpenBLAS_INCLUDE_DIR})
#    set(BLAS_LIB ${OpenBLAS_LIB})
#endif()

set(CMAKE_CXX_STANDARD 14)

add_executable(faiss_v main.cpp Index.cpp Index.h IndexFlat.cpp IndexFlat.h Heap.cpp Heap.h FaissVAssert.h FaissVException.h FaissVException.cpp utils.cpp utils.h utils_simd.cpp)