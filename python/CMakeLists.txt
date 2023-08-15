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

add_subdirectory(pybind11)

project(loopclosuretoolbox-python LANGUAGES CXX)

set(DBOW_DIRS ../src/DBoW3/include)
set(DBOW_LIBS DBoW3)
set(VLAD_DIRS ../src/VLAD/include)
set(VLAD_LIBS VLAD)
set(PYBIND_DIRS pybind11/include)

include_directories(${DBOW_DIRS} ${VLAD_DIRS} ${PYBIND_DIRS})

pybind11_add_module(loopclosuretoolbox loopclosuretoolbox.cpp)
target_link_libraries(loopclosuretoolbox PRIVATE ${DBOW_LIBS} ${VLAD_LIBS})