cmake_minimum_required(VERSION 3.0)

add_library(rodent_code STATIC
	TrackBall.cpp
	play.cpp
	rigid.cpp
)

add_executable(rodent
	main.cpp
	pyrodent.c
)

find_package(BULLET REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

find_package(PythonInterp 2 REQUIRED)
STRING( REGEX REPLACE "([0-9.]+)\\.[0-9]+" "\\1" python_version ${PYTHON_VERSION_STRING} )
find_package(PythonLibs ${python_version} REQUIRED)
find_package(NumPy REQUIRED)

if (PYTHON_NUMPY_FOUND)
	message("NumPy found")
	add_definitions(-DPYBULLET_USE_NUMPY)
else()
	message("NumPy not found")
endif()

include_directories( ${OPENGL_INCLUDE_DIRS}
					 ${GLUT_INCLUDE_DIRS}
					 ${BULLET_INCLUDE_DIRS}
					 ${PYTHON_INCLUDE_DIRS}
					 ${PYTHON_NUMPY_INCLUDE_DIR}
					 )

target_link_libraries(rodent
					${OPENGL_LIBRARIES}
					${GLUT_LIBRARY}
					${BULLET_LIBRARIES}
					${PYTHON_LIBRARIES}
					rodent_code
					)
