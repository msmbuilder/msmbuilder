cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules" ${CMAKE_MODULE_PATH})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/platforms/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/platforms/cpu/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/platforms/cpu/kernels/include")
file(GLOB_RECURSE HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/platforms/include/*.h"
                          "${CMAKE_CURRENT_SOURCE_DIR}/platforms/cpu/include/*.h"
                          "${CMAKE_CURRENT_SOURCE_DIR}/platforms/cpu/include/*.hpp"
			  "${CMAKE_CURRENT_SOURCE_DIR}/platforms/cpu/kernels/include/*.h")

file(GLOB_RECURSE CPU_KERNEL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/platforms/cpu/kernels/*.c")
file(GLOB_RECURSE CPU_PLATFORM_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/platforms/cpu/src/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/platforms/cpu/src/*.cpp")
add_library(MixtapeCPU ${CPU_PLATFORM_SOURCES} ${CPU_KERNEL_SOURCES})

file(GLOB CPU_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/platforms/cpu/tests/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/platforms/cpu/tests/*.cpp")
foreach(TEST_PROG ${CPU_TEST_SOURCES})
  get_filename_component(TEST_ROOT ${TEST_PROG} NAME_WE)
  add_executable(${TEST_ROOT} ${TEST_PROG})
  target_link_libraries(${TEST_ROOT} MixtapeCPU)
  list(APPEND LINK_TARGETS ${TEST_ROOT})
endforeach(TEST_PROG ${CPU_TEST_SOURCES})

# CUDA PLATFORM
find_package(CUDA)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/platforms/cuda/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/platforms/cuda/kernels/include")
cuda_include_directories("${CMAKE_CURRENT_SOURCE_DIR}/platforms/cuda/kernels/include")
file(GLOB_RECURSE CUDA_KERNEL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/platforms/cuda/kernels/*.cu"
                                      "${CMAKE_CURRENT_SOURCE_DIR}/platforms/cuda/src/*.cu")
cuda_add_library(MixtapeCUDA "${CUDA_KERNEL_SOURCES}")
