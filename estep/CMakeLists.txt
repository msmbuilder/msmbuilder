cmake_minimum_required(VERSION 2.6)
find_package(Boost REQUIRED)
find_package(BLAS REQUIRED)



file(GLOB_RECURSE MIXTAPE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include/")
include_directories(${Boost_INCLUDE_DIRS})


add_library (mixtape ${MIXTAPE_SOURCES})


set(SHARED_TARGET mixtape ${BLAS_LIBRARIES})
add_subdirectory(tests) 