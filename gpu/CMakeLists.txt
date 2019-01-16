# specify header and cpp files
file(GLOB_RECURSE faissv_gpu_headers ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB_RECURSE faissv_gpu_cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE faissv_gpu_cuh ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh)
file(GLOB_RECURSE faissv_gpu_cu ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

set(faissv_lib_gpu gpufaiss_v)

list(REMOVE_ITEM faissv_gpu_cpp ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_GpuIndexFlat_gpu.cpp)

cuda_add_library(${faissv_lib_gpu} STATIC ${faissv_gpu_headers} ${faissv_gpu_cpp} ${faissv_gpu_cuh} ${faissv_gpu_cu})

if(BUILD_TEST)
  add_subdirectory(tests)
endif(BUILD_TEST)

#
install(TARGETS ${faissv_lib_gpu}
    ARCHIVE DESTINATION lib
    )

install(FILES ${faissv_gpu_cuh} ${faissv_gpu_headers} DESTINATION include/gpufaiss_v)