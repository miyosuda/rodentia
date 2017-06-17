cmake_minimum_required(VERSION 3.0)

add_definitions("-Wall -std=c++11")

project(rodent)

set(PROJECT_SOURCE_DIR     ${CMAKE_BINARY_DIR}/src)
set(CMAKE_BINARY_DIR       ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH    ${CMAKE_BINARY_DIR}/lib)

include_directories("${PROJECT_SOURCE_DIR}")

add_executable(rodent
	${PROJECT_SOURCE_DIR}/main.cpp
	${PROJECT_SOURCE_DIR}/Graphics.cpp
	${PROJECT_SOURCE_DIR}/TrackBall.cpp
	${PROJECT_SOURCE_DIR}/play.cpp
	${PROJECT_SOURCE_DIR}/rigid.cpp
	)

find_package(BULLET REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

include_directories( ${OPENGL_INCLUDE_DIRS}
					 ${GLUT_INCLUDE_DIRS}
					 ${BULLET_INCLUDE_DIRS}
					 )

target_link_libraries(rodent
					${OPENGL_LIBRARIES}
					${GLUT_LIBRARY}
					${BULLET_LIBRARIES}
					)
