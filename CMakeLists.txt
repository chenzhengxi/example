cmake_minimum_required(VERSION 3.0)
if(" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "
        FATAL: In-source builds are not allowed.
        You should create a separate directory for build files.
        ")
endif()

project(example LANGUAGES CXX)

find_program (CMAKE_UNAME uname /bin /usr/bin /usr/local/bin)
if (CMAKE_UNAME)
    exec_program(uname ARGS -m OUTPUT_VARIABLE platform)
endif (CMAKE_UNAME)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

#SET(OpenCV_DIR "/usr/local/share/OpenCV")
# SET(OpenCV_DIR "/usr/local/include/opencv4")
FIND_PACKAGE(OpenCV REQUIRED)
if(OpenCV_FOUND)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
LINK_DIRECTORIES(${OpenCV_LIBRARY_DIRS})
else()
message(FATAL_ERROR "Opencv can't find")
endif (OpenCV_FOUND)

find_package(Boost COMPONENTS system date_time REQUIRED)
if(Boost_FOUND)
    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
    LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
else()
    message(FATAL_ERROR "Boost can't find")
endif()

find_package(PahoMqttCpp REQUIRED)
if(PahoMqttCpp_FOUND)
else()
    message(FATAL_ERROR "pahocpp can't find")
endif()

find_package(nlohmann_json 3.2.0 REQUIRED)

find_package(cppzmq)

find_package(PkgConfig REQUIRED)

# pkg_search_module(GSTREAMER REQUIRED gstreamer-1.0)
# find_library(LIB_GSTREAMER NAMES ${GSTREAMER_LIBRARIES} HINTS ${GSTREAMER_LIBRARY_DIRS})
# pkg_search_module(GSTREAMER_APP REQUIRED gstreamer-app-1.0) # for appsrc

# "${GSTREAMER_INCLUDE_DIRS}"
# "${GSTREAMER_APP_INCLUDE_DIRS}"
# include_directories(${CMAKE_SOURCE_DIR}/include "/usr/local/include/")
# link_directories("${GSTREAMER_LIBRARY_DIRS}")

include_directories(${PROJECT_SOURCE_DIR}/inc)
message(${PROJECT_SOURCE_DIR})
# 添加链接库查找目录，库查找序按照参数顺序从向右查
link_directories(${PROJECT_SOURCE_DIR}/libs_${platform})
message(${PROJECT_SOURCE_DIR}/libs_${platform})
# 设置生成的库文件目录，这里我们设置成和库目录相同
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/libs_${platform})
message(${LIBRARY_OUTPUT_PATH})
# 设置可执行文件目录，同样设成和库目录相同
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin_${platform})
message(${EXECUTABLE_OUTPUT_PATH})

IF(${platform} MATCHES "x86_64")
set(CMAKE_BUILD_TYPE "Debug")
ELSE()
set(CMAKE_BUILD_TYPE "Release")
ENDIF()

add_subdirectory(actor)
add_subdirectory(CzxActor)
add_subdirectory(FunixActor)
add_subdirectory(pub)
add_subdirectory(CudaTest)
add_subdirectory(TRT)
add_subdirectory(Eigen)
add_subdirectory(thrift)
add_subdirectory(ACE)

#CUDA_ADD_EXECUTABLE(${PROJECT_NAME} src/main.cpp src/client.cpp src/cvYolo.cpp src/mqtt_pub.cpp main.cpp)
add_executable(${PROJECT_NAME} src/main.cpp src/client.cpp src/cvYolo.cpp src/mqtt_pub.cpp)
add_dependencies(${PROJECT_NAME} actor CzxActor FunixActor)
target_link_libraries(${PROJECT_NAME} PRIVATE ${GSTREAMER_LIBRARIES} PRIVATE caf_core PRIVATE nlohmann_json::nlohmann_json PRIVATE PahoMqttCpp::paho-mqttpp3 paho-mqtt3as cppzmq ${OpenCV_LIBRARIES} 
CudaTest TRT testEigen test_thrift test_ace ${CMAKE_DL_LIBS})

add_subdirectory(tests)
enable_testing()
add_test(NAME ${PROJECT_NAME}_test COMMAND tests)
