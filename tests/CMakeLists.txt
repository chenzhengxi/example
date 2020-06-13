cmake_minimum_required(VERSION 3.0)

project(tests LANGUAGES CXX)

find_program (CMAKE_UNAME uname /bin /usr/bin /usr/local/bin)
if (CMAKE_UNAME)
    exec_program(uname ARGS -m OUTPUT_VARIABLE platform)
endif (CMAKE_UNAME)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Catch2 REQUIRED)
if (Catch2_FOUND)
  include_directories(${Catch2_INCLUDE_DIRS})
  #target_link_libraries (helloworld ${BZIP2_LIBRARIES})
else()
message(FATAL_ERROR "Catch2 can't find")
endif (Catch2_FOUND)

include_directories(${PROJECT_SOURCE_DIR}/include/)
message(${PROJECT_SOURCE_DIR})
# 添加链接库查找目录，库查找序按照参数顺序从向右查
link_directories(${PROJECT_SOURCE_DIR}/libs_${platform})
message(${PROJECT_SOURCE_DIR}/libs_${platform})
# 设置生成的库文件目录，这里我们设置成和库目录相同
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/libs_${platform})
message(${LIBRARY_OUTPUT_PATH})

IF(${platform} MATCHES "x86_64")
set(CMAKE_BUILD_TYPE "Debug")
ELSE()
set(CMAKE_BUILD_TYPE "Release")
ENDIF()

add_executable(${PROJECT_NAME} test.cpp test_json.cpp test_mqtt.cpp test_zmq.cpp test_actor.cpp WordStatics.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE Catch2::Catch2 PRIVATE caf_core PRIVATE nlohmann_json::nlohmann_json 
                      Boost::system actor CzxActor FunixActor cppzmq PahoMqttCpp::paho-mqttpp3 paho-mqtt3as cppzmq)

include(CTest)
include(Catch)
catch_discover_tests(${PROJECT_NAME})
