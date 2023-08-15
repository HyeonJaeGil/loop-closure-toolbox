cmake_minimum_required(VERSION 3.0)

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

project(DBoW3 VERSION 0.0.2 LANGUAGES C CXX)

# Set build flags, depending on the architecture
include_directories(include)

set(SOURCE
        src/BowVector.cpp
        src/Database.cpp
        src/DescManip.cpp
        src/FeatureVector.cpp
        src/QueryResults.cpp
        src/quicklz.c
        src/ScoringObject.cpp
        src/Vocabulary.cpp
        )
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

include(CMakePackageConfigHelpers)
set(CMAKE_INSTALL_LIBDIR lib)

set(CMAKE_EXPORT_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

add_library(${PROJECT_NAME} SHARED ${SOURCE})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES} OpenMP::OpenMP_CXX)

# install(DIRECTORY include/
#         DESTINATION third_party/include
#         COMPONENT main)

install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}Exports
        RUNTIME DESTINATION bin COMPONENT main			# Install the dll file in bin directory
        LIBRARY DESTINATION ${LIB_INSTALL_DIR} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE COMPONENT main
        ARCHIVE DESTINATION ${LIB_INSTALL_DIR} COMPONENT main)			# Install the dll.a file in lib directory


install(EXPORT ${PROJECT_NAME}Exports DESTINATION ${CMAKE_EXPORT_DESTINATION})


