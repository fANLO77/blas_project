cmake_minimum_required(VERSION 3.10)
project(BLAS_CI_Project C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)

find_package(PkgConfig REQUIRED)
pkg_check_modules(OpenBLAS REQUIRED openblas)

add_executable(perf_tests main.cpp)
target_include_directories(perf_tests PRIVATE ${OpenBLAS_INCLUDE_DIRS})
target_link_libraries(perf_tests PRIVATE ${OpenBLAS_LIBRARIES} pthread)

add_executable(interface_tests test_level2_full.c)
target_include_directories(interface_tests PRIVATE ${OpenBLAS_INCLUDE_DIRS})
target_link_libraries(interface_tests PRIVATE ${OpenBLAS_LIBRARIES} m)