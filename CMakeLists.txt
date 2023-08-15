cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(APPLE)
    # Assuming you're using LLVM/Clang installed via Homebrew on macOS
    set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
    set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")
elseif(UNIX AND NOT APPLE) # Linux
    set(CMAKE_C_COMPILER "/usr/bin/gcc")
    set(CMAKE_CXX_COMPILER "/usr/bin/g++")
endif()


option(BUILD_PYBIND "Build pybind" OFF)
option(BUILD_TEST "Build test" ON)

add_subdirectory(src/DBoW3)
add_subdirectory(src/VLAD)

IF(BUILD_PYBIND)
    add_subdirectory(python)
ENDIF(BUILD_PYBIND)

IF(BUILD_TEST)
    add_subdirectory(test)
ENDIF(BUILD_TEST)

project(loopclosuretoolbox)