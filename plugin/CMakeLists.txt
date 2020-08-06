cmake_minimum_required(VERSION 2.8)

option(Plugin_BUILD_PLUGIN_EXAMPLE "Build PluginExample" ${BUILD_ALL})


    project(PluginExample CXX)

    ##################
    #  Dependencies  #
    ##################

    find_package(Boost REQUIRED)
    mark_as_advanced(Boost_DIR)

    #######################
    #  Compilation flags  #
    #######################

    include_directories(
        ../inc
        ${Boost_INCLUDE_DIR}
    )

    ############
    #  Target  #
    ############

    add_library(${PROJECT_NAME}
        SHARED
        ${PROJECT_SOURCE_DIR}/MyPlugin.cpp
        ${PROJECT_SOURCE_DIR}/MyPlugin.h
    )

