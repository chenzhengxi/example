cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(CudaTest LANGUAGES CUDA)
#file(GLOB CXX_SRCS *.cpp)
# SET(SRC_LIST
#     CudaTest.cpp
#     )
file(GLOB CU_SRCS *.cu)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

CUDA_COMPILE(CU_OBJS ${CU_SRCS})
#add_library(CudaTest STATIC ${SRC_LIST} ${CU_OBJS})
add_library(${PROJECT_NAME} SHARED ${SRC_LIST} ${CU_OBJS})
target_link_libraries(${PROJECT_NAME} cudart cudnn cublas)