cmake_minimum_required(VERSION 3.0)

add_library(rodent_code STATIC
	TrackBall.cpp
	play.cpp
	rigid.cpp
)

add_executable(rodent
	main.cpp
)

add_library(pyrodent SHARED
	pyrodent.c
	rodent_api.cpp		
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

message(${PYTHON_NUMPY_INCLUDE_DIR})

# workaround for brew environment
#set(PYTHON_EXECUTABLE "/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/bin/python2")
set(PYTHON_INCLUDE_DIRS "/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/include/python2.7")
set(PYTHON_LIBRARIES "/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib")
set(NUMPY_INCLUDES "/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include")

#..
#message("python incude dir:" ${PYTHON_INCLUDE_DIRS})
#message("python lib dir:"    ${PYTHON_LIBRARIES})
#..

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
					rodent_code
					)

target_link_libraries(pyrodent
					${OPENGL_LIBRARIES}
					${GLUT_LIBRARY}
					${BULLET_LIBRARIES}
					${PYTHON_LIBRARIES}
					rodent_code
					)
