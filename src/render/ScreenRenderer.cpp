#include "ScreenRenderer.h"

/**
 * <!--  init():  -->
 */
bool ScreenRenderer::init(int width, int height) {
	glViewport(0, 0, width, height);
	
	float ratio = width / (float) height;
	initCamera(ratio, false);

	glDepthFunc(GL_LESS);
	glEnable(GL_DEPTH_TEST);
	
	glClearColor(0.5f, 0.5f, 0.5f, 1.0f);

	return true;
}

/**
 * <!--  renderPost():  -->
 */
void ScreenRenderer::renderPost() {
}

/**
 * <!--  release():  -->
 */
void ScreenRenderer::release() {
}
