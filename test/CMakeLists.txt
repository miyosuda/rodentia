cmake_minimum_required(VERSION 3.0)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(GTEST REQUIRED)

include_directories( ${GTEST_ROOT}/include/ )
link_directories( ${GTEST_ROOT}/lib/ )

add_executable(rodent_test
	${CMAKE_CURRENT_SOURCE_DIR}/matrix_test.cpp
)

target_link_libraries( rodent_test
	rodent_code
	gtest
	gtest_main
	pthread
)
