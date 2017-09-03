#include "Renderer.h"

#include <stdlib.h>
#include <stdio.h>

#include "glinc.h"


/**
 * <!--  init():  -->
 */
bool Renderer::init(int width, int height) {
	bool ret;

	ret = context.init(width, height);
	
	if( !ret ) {
		printf("Failed to initialize GL context\n");
		return false;
	}

	ret = gladLoadGL();

	if( !ret ) {
		printf("Failed to init glad.\n");
		return false;
	}

	frameBufferWidth = width;
	frameBufferHeight = height;

	ret = frameBuffer.init(frameBufferWidth, frameBufferHeight);
	
	if( !ret ) {
		printf("Failed to init offscreen frame buffer.\n");
		return false;
	}

	buffer = calloc(4, getFrameBufferSize()/4);
	
	return true;
}

/**
 * <!--  release():  -->
 */
void Renderer::release() {
	free(buffer);
	buffer = nullptr;
	context.release();
}

/**
 * <!--  prepareRendering():  -->
 */
void Renderer::prepareRendering() {
	frameBuffer.use();
	
	glViewport(0, 0, frameBufferWidth, frameBufferHeight);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glDepthFunc(GL_LESS);
	glEnable(GL_DEPTH_TEST);
	
	glClearColor(0.54f, 0.80f, 0.98f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

/**
 * <!--  finishRendering():  -->
 */
void Renderer::finishRendering() {
	glReadPixels(0, 0, frameBufferWidth, frameBufferHeight, GL_RGB, GL_UNSIGNED_BYTE, buffer);
}
