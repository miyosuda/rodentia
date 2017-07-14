cmake_minimum_required(VERSION 3.0)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(GTEST REQUIRED)
find_package(OpenGL REQUIRED)

include_directories( ${GTEST_ROOT}/include/ )
include_directories( ${GTEST_ROOT}/include/
					 ../src/common
					 ../src/geom
					 ../src/play
					 ../src/render
					 ../src/env
)
link_directories( ${GTEST_ROOT}/lib/ )

add_executable(rodent_test
	${CMAKE_CURRENT_SOURCE_DIR}/matrix_test.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/mesh_manager_test.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/mesh_face_data_test.cpp
)

target_link_libraries( rodent_test
    ${OPENGL_LIBRARIES}
	rodent_code
	gtest
	gtest_main
	pthread
)
