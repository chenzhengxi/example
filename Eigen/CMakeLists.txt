cmake_minimum_required(VERSION 3.0)
if(" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "
        FATAL: In-source builds are not allowed.
        You should create a separate directory for build files.
        ")
endif()

project(testEigen LANGUAGES CXX)


set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
#add_definitions("-DENABLE_SSE")

find_package(Eigen3)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIRS})
message(${EIGEN3_INCLUDE_DIRS})

find_package(Sophus)


find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

IF(${platform} MATCHES "x86_64")
set(CMAKE_BUILD_TYPE "Debug")
ELSE()
set(CMAKE_BUILD_TYPE "Release")
ENDIF()

add_library(${PROJECT_NAME} SHARED "testEigen.cpp" "Hungarian.cpp")
target_link_libraries(${PROJECT_NAME} Sophus::Sophus ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
