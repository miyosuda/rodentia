cmake_minimum_required(VERSION 3.0)

SET(CMAKE_CXX_FLAGS "-Wall -std=c++11")

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(OpenGL REQUIRED)
find_package(PNG REQUIRED)

include_directories( ${GTEST_ROOT}/include/ )
include_directories( ../src/common
					 ../src/geom
					 ../src/play
					 ../src/render
					 ../src/env
					 ../third_party/glad/include
					 ../third_party/bullet3-2.86.1/src/
)

add_executable(rodent_test
	${CMAKE_CURRENT_SOURCE_DIR}/matrix_test.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/bounding_box_test.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/camera_test.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/environment_object_info_test.cpp
	../third_party/glad/src/glad.c
)

set(BULLET_LIBRARIES
	BulletInverseDynamics
    BulletDynamics
	BulletSoftBody
    BulletCollision
	LinearMath  
	Bullet3Collision
	Bullet3Dynamics
	Bullet3Common
	Bullet3Geometry
)

target_link_libraries( rodent_test
	rodent_code
	gtest
	gtest_main
	pthread
	dl
	${PNG_LIBRARIES}	
    ${OPENGL_LIBRARIES}
	${BULLET_LIBRARIES}
)
