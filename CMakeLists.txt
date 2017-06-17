cmake_minimum_required(VERSION 3.0)

add_definitions("-Wall -std=c++11")
add_definitions("-Wno-deprecated-declarations")

project(rodent)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)
add_subdirectory(test)


