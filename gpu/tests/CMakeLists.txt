file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Build each source file independently
include_directories(../../)	# faiss root directory

# gtest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

foreach(source ${srcs})
    get_filename_component(name ${source} NAME_WE)

    # target
    add_executable(${name} ${source})
    target_link_libraries(${name} ${faissv_lib_gpu} ${faissv_lib} ${GTEST_BOTH_LIBRARIES} ${CUDA_LINKER_LIBS})

    # Install
    install(TARGETS ${name} DESTINATION test)
endforeach(source)
