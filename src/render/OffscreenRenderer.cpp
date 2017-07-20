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

	// Set GL version
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

	// RGB 24bit format (No alpha)
	glfwWindowHint(GLFW_RED_BITS, 8);
	glfwWindowHint(GLFW_GREEN_BITS, 8);
	glfwWindowHint(GLFW_BLUE_BITS, 8);
	
	// TODO: Tone jump occurs when explicitly set zero alpha bits.
	//glfwWindowHint(GLFW_ALPHA_BITS, 0);

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

	float ratio = width / (float) height;
	initCamera(ratio, true);

	glDepthFunc(GL_LESS);
	glEnable(GL_DEPTH_TEST);
	
	glClearColor(0.5f, 0.5f, 0.5f, 1.0f);

#if !USE_NATIVE_OSMESA
	buffer = calloc(4, getFrameBufferSize()/4);
#endif

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
 * <!--  renderPost():  -->
 */
void OffscreenRenderer::renderPost() {
#if USE_NATIVE_OSMESA
	glfwGetOSMesaColorBuffer(window, &frameBufferWidth, &frameBufferHeight,
							 NULL, &buffer);
#else
	glReadPixels(0, 0, frameBufferWidth, frameBufferHeight, GL_RGB, GL_UNSIGNED_BYTE, buffer);
#endif
}
