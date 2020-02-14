cmake_minimum_required(VERSION 3.0)

project(rodentia)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

# Add cmake_modules to module
set(CMAKE_MODULE_PATH
	${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}"
	"${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules")

add_subdirectory(third_party)
add_subdirectory(src)
add_subdirectory(test)
