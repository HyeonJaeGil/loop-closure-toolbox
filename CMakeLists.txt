cmake_minimum_required(VERSION 3.0.0)
project(VLAD)

add_subdirectory(src/DBoW3)
add_subdirectory(src/VLAD)
add_subdirectory(python)

IF(BUILD_TEST)
    add_subdirectory(test)
ENDIF(BUILD_TEST)