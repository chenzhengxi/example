project(test_thrift LANGUAGES CXX)

#Make sure gen-cpp files can be included
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/gen-cpp")
include_directories("${PROJECT_SOURCE_DIR}/lib/cpp/src")


set(tutorialgencpp_SOURCES
    ${PROJECT_SOURCE_DIR}/gen-cpp/Calculator.cpp
    ${PROJECT_SOURCE_DIR}/gen-cpp/SharedService.cpp
    ${PROJECT_SOURCE_DIR}/gen-cpp/shared_constants.cpp
    ${PROJECT_SOURCE_DIR}/gen-cpp/shared_types.cpp
    ${PROJECT_SOURCE_DIR}/gen-cpp/tutorial_constants.cpp
    ${PROJECT_SOURCE_DIR}/gen-cpp/tutorial_types.cpp
)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
find_package(Thrift)
if(Thrift_FOUND)
    INCLUDE_DIRECTORIES(${Thrift_INCLUDE_DIRS})
    LINK_DIRECTORIES(${Thrift_LIBRARY_DIRS})
else()
    message(FATAL_ERROR "thrift can't find")
endif()

add_library(tutorialgencpp STATIC ${tutorialgencpp_SOURCES})
target_link_libraries(tutorialgencpp thrift)

add_custom_command(OUTPUT 
                  ${tutorialgencpp_SOURCES}
    COMMAND thrift --gen cpp -out ${PROJECT_SOURCE_DIR}/gen-cpp -r ${PROJECT_SOURCE_DIR}/tutorial.thrift
    COMMAND rm -f ${PROJECT_SOURCE_DIR}/gen-cpp/Calculator_server.skeleton.cpp
    COMMAND rm -f ${PROJECT_SOURCE_DIR}/gen-cpp/SharedService_server.skeleton.cpp
    DEPENDS ${PROJECT_SOURCE_DIR}/tutorial.thrift
    COMMENT "Now Generating thrift files.............................."
)

IF(${platform} MATCHES "x86_64")
set(CMAKE_BUILD_TYPE "Debug")
ELSE()
set(CMAKE_BUILD_TYPE "Release")
ENDIF()

add_library(${PROJECT_NAME} SHARED test_thrift.cpp)
target_link_libraries(${PROJECT_NAME} tutorialgencpp)
target_link_libraries(${PROJECT_NAME} thrift)
if (ZLIB_FOUND)
  target_link_libraries(${PROJECT_NAME} ${ZLIB_LIBRARIES})
endif ()
