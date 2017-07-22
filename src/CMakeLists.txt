cmake_minimum_required(VERSION 3.0)

SET(CMAKE_CXX_FLAGS "-Wall -std=c++11")

option(BUILD_PYTHON3 "Build Rodent Python3 binding instead of default Python2." OFF)

# OS-dependent specifics
if(APPLE)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
endif()

add_library(rodent_code STATIC
	play/TrackBall.cpp
	play/play.cpp
	env/Environment.cpp
	env/RigidBodyComponent.cpp
	env/EnvironmentObject.cpp
	render/DebugDrawer.cpp
	render/Renderer.cpp
	render/OffscreenRenderer.cpp
	render/ScreenRenderer.cpp
	render/DiffuseShader.cpp
	render/Image.cpp
	render/Camera.cpp
	render/LineShader.cpp
	render/Material.cpp
	render/Mesh.cpp
	render/MeshFace.cpp
	render/MeshFaceData.cpp
	render/MeshManager.cpp
	render/OffscreenRenderer.cpp
	render/PNGDecoder.cpp
	render/Renderer.cpp
	render/ScreenRenderer.cpp
	render/Shader.cpp
	render/ShaderManager.cpp
	render/Texture.cpp
	render/TextureManager.cpp
	render/DrawComponent.cpp
)

add_executable(rodent
	play/main.cpp
)

add_library(rodent_module SHARED
	python/rodent_module.cpp
)

set_target_properties(rodent_module PROPERTIES PREFIX "")
set_target_properties(rodent_module PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/rodent)


include_directories(common geom play render env)

find_package(BULLET REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLFW REQUIRED)
find_package(PNG REQUIRED)

if (BUILD_PYTHON3)

	find_package(PythonInterp 3 REQUIRED)
	STRING(REGEX REPLACE "([0-9.]+)\\.[0-9]+" "\\1" python3_version ${PYTHON_VERSION_STRING})
	find_package(PythonLibs ${python3_version} REQUIRED)
	
	find_package(NumPy REQUIRED)

	message(STATUS "aaa") #..
	message(STATUS ${NUMPY_INCLUDES}) #..
	message(STATUS "bbb") #..

else ()
	
	find_package(PythonInterp 2 REQUIRED)
	STRING( REGEX REPLACE "([0-9.]+)\\.[0-9]+" "\\1" python_version ${PYTHON_VERSION_STRING} )
	find_package(PythonLibs ${python_version} REQUIRED)
	find_package(NumPy REQUIRED)
	
	if (PYTHON_NUMPY_FOUND)
		message("NumPy found")
		add_definitions(-DUSE_NUMPY)
	else()
		message("NumPy not found")
	endif()

	# TODO: workaround for brew environment
	set(PYTHON_INCLUDE_DIRS "/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/include/python2.7")
	set(PYTHON_LIBRARIES "/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib")
	set(NUMPY_INCLUDES "/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include")

endif ()

include_directories( ${OPENGL_INCLUDE_DIRS}
					 ${GLFW_INCLUDE_DIRS}
					 ${BULLET_INCLUDE_DIRS}
					 ${PNG_INCLUDE_DIRS}
					 ${PYTHON_INCLUDE_DIRS}					 
					 ${NUMPY_INCLUDES}
					 )

target_link_libraries(rodent
					${OPENGL_LIBRARIES}
					${GLFW_LIBRARIES}
					${BULLET_LIBRARIES}
					${PNG_LIBRARIES}
					rodent_code
					)

target_link_libraries(rodent_module
					${OPENGL_LIBRARIES}
					${GLFW_LIBRARIES}
					${BULLET_LIBRARIES}
					${PNG_LIBRARIES}
					${PYTHON_LIBRARIES}
					rodent_code
					)
