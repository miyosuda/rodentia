### GLFW ###

add_subdirectory(glfw)

include_directories(
	glfw/include/GLFW/
)

# TODO: check options below

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
set(OPENGL_LIBRARY
	${OPENGL_LIBRARY}
	-lGL -lGLU -lXrandr -lXext -lX11 -lrt
	${CMAKE_DL_LIBS}
	${GLFW_LIBRARIES}
)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
set(OPENGL_LIBRARY
	${OPENGL_LIBRARY}
	${CMAKE_DL_LIBS}
	${GLFW_LIBRARIES}
)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")


### BULLET ###

set(BULLET_VERSION 2.86.1)

include_directories(
	bullet3-2.86.1/src
)
add_subdirectory( bullet3-2.86.1/src/Bullet3Collision )
add_subdirectory( bullet3-2.86.1/src/Bullet3Common )
add_subdirectory( bullet3-2.86.1/src/Bullet3Dynamics )
add_subdirectory( bullet3-2.86.1/src/Bullet3Geometry )
add_subdirectory( bullet3-2.86.1/src/Bullet3Serialize/Bullet2FileLoader )
add_subdirectory( bullet3-2.86.1/src/BulletCollision )
add_subdirectory( bullet3-2.86.1/src/BulletDynamics )
add_subdirectory( bullet3-2.86.1/src/BulletInverseDynamics )
add_subdirectory( bullet3-2.86.1/src/BulletSoftBody )
add_subdirectory( bullet3-2.86.1/src/LinearMath )


### GTEST ###

add_subdirectory( googletest )
