cmake_minimum_required(VERSION 2.8)

add_definitions("-Wall -std=c++11")

project(rodent)

set(BULLET_PATH "../bullet3-2.86.1/src")

set(PROJECT_SOURCE_DIR     ${CMAKE_BINARY_DIR}/src)
set(CMAKE_BINARY_DIR       ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH    ${CMAKE_BINARY_DIR}/lib)

include_directories("${PROJECT_SOURCE_DIR}")

#..
link_directories("${BULLET_PATH}/Bullet3Collision")
link_directories("${BULLET_PATH}/Bullet3Common")
link_directories("${BULLET_PATH}/Bullet3Dynamics")
link_directories("${BULLET_PATH}/Bullet3Geometry")
link_directories("${BULLET_PATH}/Bullet3OpenCL")
link_directories("${BULLET_PATH}/BulletCollision")
link_directories("${BULLET_PATH}/BulletDynamics")
link_directories("${BULLET_PATH}/BulletInverseDynamics")
link_directories("${BULLET_PATH}/BulletSoftBody")
link_directories("${BULLET_PATH}/LinearMath")
#..

add_executable(rodent
	${PROJECT_SOURCE_DIR}/main.cpp
	${PROJECT_SOURCE_DIR}/Graphics.cpp
	${PROJECT_SOURCE_DIR}/TrackBall.cpp
	${PROJECT_SOURCE_DIR}/play.cpp
	${PROJECT_SOURCE_DIR}/rigid.cpp
	)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS} )
target_link_libraries(rodent ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} )

include_directories( ${BULLET_PATH} )


target_link_libraries(rodent
	Bullet3Collision
	Bullet3Common
	Bullet3Dynamics
	Bullet3Geometry
	Bullet3OpenCL_clew
	BulletCollision
	BulletDynamics
	BulletInverseDynamics
	BulletSoftBody
	LinearMath
	)


