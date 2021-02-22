cmake_minimum_required(VERSION 3.0)


SET(CMAKE_CXX_FLAGS "-Wall -std=c++11 -fPIC")
SET(CMAKE_C_FLAGS "-Wall -fPIC")

# OS-dependent specifics
if(APPLE)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
endif()

add_library(rodentia_code STATIC
	env/Environment.cpp
	env/RigidBodyComponent.cpp
	env/EnvironmentObject.cpp
    env/CollisionShapeManager.cpp
	render/DebugDrawer.cpp
	render/DiffuseShader.cpp
	render/Image.cpp
	render/Camera.cpp
	render/LineShader.cpp
	render/Material.cpp
	render/Mesh.cpp
	render/MeshFace.cpp
	render/MeshFaceData.cpp
	render/MeshData.cpp		
	render/MeshManager.cpp
	render/PNGDecoder.cpp
	render/Shader.cpp
	render/ShaderManager.cpp
	render/Texture.cpp
	render/TextureManager.cpp
	render/DrawComponent.cpp
	render/RenderingContext.cpp
	render/OffscreenFrameBuffer.cpp
	render/GLContext.cpp
	render/DepthFrameBuffer.cpp
	render/ShadowDepthShader.cpp
	render/ShadowDiffuseShader.cpp
	render/LSPSM.cpp
    render/RenderTarget.cpp
    render/CameraView.cpp
	importer/ObjImporter.cpp
	geom/BoundingBox.cpp
)

add_library(rodentia_module SHARED
	python/rodentia_module.cpp
	../third_party/glad/src/glad.c
)

set_target_properties(rodentia_module PROPERTIES PREFIX "")
set_target_properties(rodentia_module PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/rodentia)


include_directories(common geom play render env importer
   ../third_party/glad/include
   ../third_party/bullet3-2.86.1/src/
   ../third_party/libpng-1.6.37/
   ${CMAKE_BINARY_DIR}/third_party/libpng-1.6.37/
)

find_package(OpenGL REQUIRED)

find_package(PythonInterp 3 REQUIRED)
STRING(REGEX REPLACE "([0-9.]+)\\.[0-9]+" "\\1"
	   PYTHON3_SHORT_VERSION ${PYTHON_VERSION_STRING})
find_package(PythonLibs ${PYTHON3_SHORT_VERSION} REQUIRED)
	
find_package(NumPy REQUIRED)

include_directories( ${OPENGL_INCLUDE_DIRS}
					 ${PYTHON_INCLUDE_DIRS}
					 ${NUMPY_INCLUDES}
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

set(PNG_LIBRARIES
    png_static
)

target_link_libraries(rodentia_module
					rodentia_code
					${OPENGL_LIBRARIES}
					${BULLET_LIBRARIES}
					${PYTHON_LIBRARIES}
                    ${PNG_LIBRARIES}
					)
