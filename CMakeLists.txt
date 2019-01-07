cmake_minimum_required(VERSION 2.8.7)

project(faiss_v C CXX)

option(BUILD_TUTORIAL "Build tutorials" ON)
option(BUILD_TEST "Build tests" ON)
option(BUILD_WITH_GPU "Build faiss with gpu (cuda) support" ON)
option(WITH_MKL "Build with MKL if ON (OpenBLAS if OFF)" OFF)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# OpenMP
find_package(OpenMP REQUIRED)

# BLAS (MKL os OpenBLAS)
if(WITH_MKL)
    message("WITH_MKL")
    find_package(MKL REQUIRED)
    include_directories(${MKL_INCLUDE_DIRS})
    set(BLAS_LIB ${MKL_LIBRARIES})
else()
    message("BLAS")
    find_package(OpenBLAS REQUIRED)
    include_directories(${OpenBLAS_INCLUDE_DIR})
    set(BLAS_LIB ${OpenBLAS_LIB})
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -m64 -Wall -g -O3  -msse4 -mpopcnt -fopenmp -Wno-sign-compare")
add_definitions(-DFINTEGER=int)

# specify output bin_path and lib_path
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# specify header and cpp files
file(GLOB faiss_cpu_headers ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB faiss_cpu_cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
list(REMOVE_ITEM faiss_cpu_cpp  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
message(${faiss_cpu_cpp})

set(faiss_lib faiss_v)
#add_executable(${faiss_lib} ${faiss_cpu_headers} ${faiss_cpu_cpp})
add_library(${faiss_lib} STATIC ${faiss_cpu_headers} ${faiss_cpu_cpp})
target_link_libraries(${faiss_lib} ${OpenMP_CXX_FLAGS} ${BLAS_LIB})

# build gpu lib
if(BUILD_WITH_GPU)
    include(cmake/Cuda.cmake)
    add_subdirectory(gpu)
endif(BUILD_WITH_GPU)

# build tests
if(BUILD_TEST)
    add_subdirectory(tests)
endif(BUILD_TEST)
# Install libraries
install(TARGETS ${faiss_lib}
    ARCHIVE DESTINATION lib
)
install(FILES ${faiss_cpu_headers} DESTINATION include/faiss_v)

