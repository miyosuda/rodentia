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

	// Setup rendering frame buffer
	ret = frameBuffer.init(frameBufferWidth, frameBufferHeight);
	
	if( !ret ) {
		printf("Failed to init offscreen frame buffer.\n");
		return false;
	}

	// Setup shadow depth frame buffer
	// TODO: バッファサイズ調整
	ret = depthFrameBuffer.init(1024, 1024);
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

	//glFrontFace(GL_CW); // flipped because camera is inverted
	//glEnable(GL_CULL_FACE);
	//glCullFace(GL_BACK);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
}

/**
 * <!--  prepareRendering():  -->
 */
void Renderer::prepareRendering() {
	frameBuffer.use();
	
	frameBuffer.setViewport();

	glFrontFace(GL_CW); // flipped because camera is inverted
	
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	
	glClearColor(0.54f, 0.80f, 0.98f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//..
	glEnable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE1);
	depthFrameBuffer.bind();
	//..
}

/**
 * <!--  finishRendering():  -->
 */
void Renderer::finishRendering() {
	glReadPixels(0, 0, frameBufferWidth, frameBufferHeight, GL_RGB, GL_UNSIGNED_BYTE, buffer);
}
