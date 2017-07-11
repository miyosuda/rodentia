cmake_minimum_required(VERSION 2.8)

add_definitions("-Wall -std=c++11")

project(glcheck)

set(PROJECT_SOURCE_DIR     ${CMAKE_BINARY_DIR}/src)
set(CMAKE_BINARY_DIR       ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH    ${CMAKE_BINARY_DIR}/lib)

# Add cmake_modules to module
set(CMAKE_MODULE_PATH
	${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}"
	"${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules")

find_package(OpenGL REQUIRED)
find_package(GLFW REQUIRED)
find_package(PNG REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}
					${GLFW_INCLUDE_DIRS}
					${PNG_INCLUDE_DIRS}
					)

add_executable(glcheck
	${PROJECT_SOURCE_DIR}/main.cpp
	${PROJECT_SOURCE_DIR}/Shader.cpp
	${PROJECT_SOURCE_DIR}/DiffuseShader.cpp
	${PROJECT_SOURCE_DIR}/Camera.cpp
	${PROJECT_SOURCE_DIR}/Texture.cpp
	${PROJECT_SOURCE_DIR}/PNGDecoder.cpp
	${PROJECT_SOURCE_DIR}/Image.cpp
	${PROJECT_SOURCE_DIR}/MeshFace.cpp
	${PROJECT_SOURCE_DIR}/Material.cpp
   )

target_link_libraries(glcheck
					${OPENGL_LIBRARIES}
					${GLFW_LIBRARIES}
					${PNG_LIBRARIES}
					)
