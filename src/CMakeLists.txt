cmake_minimum_required(VERSION 3.0)

add_library(rodent_code STATIC
	TrackBall.cpp
	play.cpp
	rigid.cpp
)

add_executable(rodent
	main.cpp
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
					rodent_code
					)
