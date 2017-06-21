#include "ScreenRenderer.h"

/**
 * <!--  init():  -->
 */
bool ScreenRenderer::init(int width, int height) {
	glViewport(0, 0, width, height);
	
	setProjection(width, height);

	glDepthFunc(GL_LESS);
	glEnable(GL_DEPTH_TEST);
	
	glClearColor(0.5f, 0.5f, 0.5f, 1.0f);

	return true;
}

/**
 * <!--  render():  -->
 */
void ScreenRenderer::render() {
	// TODO: 場所ここでない方がいいか？
	drawFloor();
}

/**
 * <!--  release():  -->
 */
void ScreenRenderer::release() {
}
