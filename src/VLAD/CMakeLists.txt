cmake_minimum_required(VERSION 2.18...3.16)
project(VLAD)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif ()

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


#################################
#
#          OpenCV >= 3.3.1
#
#################################
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    if (OpenCV_VERSION VERSION_LESS "3.3.1")
        message(FATAL_ERROR "OpenCV >= 3.3.1 not found")
    else ()
        message(STATUS "OpenCV ${OpenCV_VERSION} found.")
    endif ()
else ()
    message(FATAL_ERROR "OpenCV >= 3.3.1 not found")
endif ()

#################################
#
#             DBoW3
#
#################################
set(DBOW_DIRS ../DBoW3/include)
set(DBOW_LIBS DBoW3)

#################################
#
#             Eigen3
#
#################################
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 ${Eigen3_VERSION} found.")

#################################
#
#             OpenMP
#
#################################
find_package(OpenMP)
if (OpenMP_FOUND)
    message(STATUS "OpenMP found.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else ()
    message(STATUS "OpenMP not found")
endif ()

#################################
#
#        Install Library
#
#################################


if (CMAKE_BUILD_TYPE MATCHES Release)
    message("Release Mode")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
elseif (CMAKE_BUILD_TYPE MATCHES Debug)
    message("Debug Mode")
    add_definitions(-DDEBUG -DDEBUGMSF)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
endif ()

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${DBOW_DIRS}
)

file(GLOB SOURCE "src/*.cpp")

add_library(${PROJECT_NAME} SHARED ${SOURCE})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${DBOW_LIBS} OpenMP::OpenMP_CXX)
