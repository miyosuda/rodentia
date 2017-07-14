cmake_minimum_required(VERSION 3.0)

add_library(rodent_code STATIC
	play/TrackBall.cpp
	play/play.cpp
	env/Environment.cpp
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
)

add_executable(rodent
	play/main.cpp
)

add_library(pyrodent SHARED
	python/pyrodent.cpp
)

include_directories(common geom play render env)

find_package(BULLET REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLFW REQUIRED)

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

message(STATUS ${PYTHON_NUMPY_INCLUDE_DIR})

# workaround for brew environment
set(PYTHON_INCLUDE_DIRS "/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/include/python2.7")
set(PYTHON_LIBRARIES "/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib")
set(PYTHON_NUMPY_INCLUDE_DIR "/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include")

include_directories( ${OPENGL_INCLUDE_DIRS}
					 ${GLFW_INCLUDE_DIRS}
					 ${BULLET_INCLUDE_DIRS}
					 ${PYTHON_INCLUDE_DIRS}
					 ${PYTHON_NUMPY_INCLUDE_DIR}
					 )

target_link_libraries(rodent
					${OPENGL_LIBRARIES}
					${GLFW_LIBRARIES}
					${BULLET_LIBRARIES}
					rodent_code
					)

target_link_libraries(pyrodent
					  ${OPENGL_LIBRARIES}
					${GLFW_LIBRARIES}
					${BULLET_LIBRARIES}
					${PYTHON_LIBRARIES}
					rodent_code
					)
