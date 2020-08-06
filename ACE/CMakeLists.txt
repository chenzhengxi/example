cmake_minimum_required(VERSION 3.0)
if(" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "
        FATAL: In-source builds are not allowed.
        You should create a separate directory for build files.
        ")
endif()

project(test_ace LANGUAGES CXX)

find_program (CMAKE_UNAME uname /bin /usr/bin /usr/local/bin)
if (CMAKE_UNAME)
    exec_program(uname ARGS -m OUTPUT_VARIABLE platform)
endif (CMAKE_UNAME)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_search_module(ACE REQUIRED ACE)
MESSAGE(STATUS "ace dirs:" ${ACE_INCLUDE_DIRS})
MESSAGE(STATUS "ace lib:" ${ACE_LIBRARIES})
include_directories(${ACE_INCLUDE_DIRS})
link_directories(${ACE_LIBRARIES})

IF(${platform} MATCHES "x86_64")
set(CMAKE_BUILD_TYPE "Debug")
ELSE()
set(CMAKE_BUILD_TYPE "Release")
ENDIF()

add_library(${PROJECT_NAME} SHARED test_ace.cpp)
target_link_libraries(${PROJECT_NAME} ACE)