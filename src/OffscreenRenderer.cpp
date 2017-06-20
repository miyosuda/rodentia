#include "OffscreenRenderer.h"
#include <stdlib.h>

/**
 * <!--  init():  -->
 */
bool OffscreenRenderer::init(int width, int height) {
	auto renderErrorCallback = [](int error, const char* description) {
		fprintf(stderr, "Error: %s\n", description);
	};
	
	glfwSetErrorCallback(renderErrorCallback);

	if (!glfwInit()) {
		return false;
	}

	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

#if defined(__APPLE__)
	// Workaround for retina MacBook Pro
	// (MacOSX frame buffer size is doubled (because of retina screen?)
	int windowWidth  = width/2;
	int windowHeight = height/2;
#else
	int windowWidth  = width;
	int windowHeight = height;
#endif

	window = glfwCreateWindow(windowWidth, windowHeight, "", NULL, NULL);
	if (!window) {
		glfwTerminate();
		return false;
	}

	glfwMakeContextCurrent(window);
	
	glfwGetFramebufferSize(window, &frameBufferWidth, &frameBufferHeight);
	
	glViewport(0, 0, frameBufferWidth, frameBufferHeight);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	setProjection(frameBufferWidth, frameBufferHeight);

	glDepthFunc(GL_LESS);
	glEnable(GL_DEPTH_TEST);
	
	glClearColor(0.5f, 0.5f, 0.5f, 1.0f);

#if !USE_NATIVE_OSMESA
	buffer = calloc(4, getFrameBufferSize()/4);
#endif

	camera.setIdentity();

	//..
	// TODO: set initial camera matrix properly
	Matrix4f mat;
	mat.setIdentity();
	mat.setColumn(3, Vector4f(0.0f, 2.0f, 7.0f, 1.0f));
	setCamera(mat);
	//..
	
	return true;
}

void OffscreenRenderer::release() {
#if !USE_NATIVE_OSMESA
	free(buffer);
#endif
	
	glfwDestroyWindow(window);
	window = nullptr;
	
	glfwTerminate();
}

/**
 * <!--  render():  -->
 */
void OffscreenRenderer::render() {
	// TODO: 場所ここでない方がいいか？
	drawFloor();
	
#if USE_NATIVE_OSMESA
	glfwGetOSMesaColorBuffer(window, &frameBufferWidth, &frameBufferHeight,
							 NULL, &buffer);
#else
	glReadPixels(0, 0, frameBufferWidth, frameBufferHeight, GL_RGBA, GL_UNSIGNED_BYTE, buffer);
#endif
}
