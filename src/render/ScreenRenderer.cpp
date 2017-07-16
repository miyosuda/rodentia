#include "ScreenRenderer.h"

/**
 * <!--  init():  -->
 */
bool ScreenRenderer::init(int width, int height) {
	glViewport(0, 0, width, height);
	
	float ratio = width / (float) height;
	camera.init(1.0f, 1000.0f, 50.0f, ratio, false); 

	glDepthFunc(GL_LESS);
	glEnable(GL_DEPTH_TEST);
	
	glClearColor(0.5f, 0.5f, 0.5f, 1.0f);

	return true;
}

/**
 * <!--  render():  -->
 */
void ScreenRenderer::render() {
	// TODO: 整理
}

/**
 * <!--  release():  -->
 */
void ScreenRenderer::release() {
}
