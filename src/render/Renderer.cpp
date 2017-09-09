#include "Renderer.h"

#include <stdlib.h>
#include <stdio.h>

#include "glinc.h"

/**
 * <!--  calcDepthFrameBufferSize():  -->
 *
 * Calculate size of depth buffer for shadow mapping based on view width and height.
 */
int Renderer::calcDepthFrameBufferSize(int width, int height) {
	int minSize = width;
	if( height < minSize ) {
		minSize = height;
	}

	int result = 64;
	while(true) {
		if( result >= minSize ) {
			break;
		}
		result *= 2;
	}
	if( result > 1024 ) {
		// Max shadow mapp depth buffer size is 1024x1024 
		result = 1024;
	}

	return result;
}


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

	// Setup rendering frame buffer
	ret = frameBuffer.init(frameBufferWidth, frameBufferHeight);
	
	if( !ret ) {
		printf("Failed to init offscreen frame buffer.\n");
		return false;
	}

	// Setup shadow depth frame buffer
	int depthFrameBufferSize = calcDepthFrameBufferSize(width, height);
	ret = depthFrameBuffer.init(depthFrameBufferSize, depthFrameBufferSize);
	if( !ret ) {
		printf("Failed to init shadow depth buffer.\n");
		return false;
	}
	
	buffer = calloc(4, getFrameBufferSize()/4);
	
	return true;
}

/**
 * <!--  release():  -->
 */
void Renderer::release() {
	if( buffer != nullptr ) {
		free(buffer);
		buffer = nullptr;
	}
	context.release();
}

/**
 * <!--  prepareShadowDepthRendering():  -->
 */
void Renderer::prepareShadowDepthRendering() {
	depthFrameBuffer.use();
	
	depthFrameBuffer.setViewport();

	glFrontFace(GL_CCW); // Default
	glEnable(GL_CULL_FACE); 
	glCullFace(GL_FRONT); // Cull front face
	
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	
	glClear(GL_DEPTH_BUFFER_BIT);
}

/**
 * <!--  prepareRendering():  -->
 */
void Renderer::prepareRendering() {
	frameBuffer.use();
	
	frameBuffer.setViewport();

	glFrontFace(GL_CW); // Flipped because camera is inverted as upside down
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK); // Cull back face
	
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	
	//glClearColor(0.54f, 0.80f, 0.98f, 1.0f);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Set depth frame buffer for texture slot 1
	glEnable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE1);
	depthFrameBuffer.bind();
}

/**
 * <!--  finishRendering():  -->
 */
void Renderer::finishRendering() {
	glReadPixels(0, 0, frameBufferWidth, frameBufferHeight, GL_RGB, GL_UNSIGNED_BYTE, buffer);
}
