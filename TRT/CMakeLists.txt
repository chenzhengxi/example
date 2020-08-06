cmake_minimum_required(VERSION 3.0)
if(" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "
        FATAL: In-source builds are not allowed.
        You should create a separate directory for build files.
        ")
endif()

project(TRT LANGUAGES CXX)

SET(SRC_LIST
    sampleOnnxMNIST.cpp
    common/logger.cpp
    common/sampleEngines.cpp
    common/sampleOptions.cpp
    )

find_package(OpenCV REQUIRED core imgproc imgcodecs highgui REQUIRED)

find_package(CUDA REQUIRED cudart cublas curand REQUIRED)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
include_directories(${CUDA_INCLUDE_DIRS} ./common)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# CUDA_COMPILE(CU_OBJS ${CU_SRCS})
#add_library(CudaTest STATIC ${SRC_LIST} ${CU_OBJS})
add_library(${PROJECT_NAME} SHARED ${SRC_LIST})
target_link_libraries(${PROJECT_NAME} cudart cudnn cublas nvinfer nvonnxparser nvparsers nvinfer_plugin)
