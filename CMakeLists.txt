cmake_minimum_required(VERSION 3.12)
project(VLAD)

option(BUILD_PYBIND "Build pybind" ON)
option(BUILD_TEST "Build test" OFF)

add_subdirectory(src/DBoW3)
add_subdirectory(src/VLAD)

IF(BUILD_PYBIND)
    add_subdirectory(python)
ENDIF(BUILD_PYBIND)

IF(BUILD_TEST)
    add_subdirectory(test)
ENDIF(BUILD_TEST)