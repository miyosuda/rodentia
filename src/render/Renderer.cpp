#include "Renderer.h"
#include "glinc.h"

/**
 * <!--  initCamera():  -->
 */
void Renderer::initCamera(float ratio, bool flipping) {
	const float nearClip = 1.0f;
	const float farClip = 1000.0f;
	const float focalLength = 50.0f;
	
	camera.init(nearClip, farClip, focalLength, ratio, flipping);
}

/**
 * <!--  setCameraMat():  -->
 */
void Renderer::setCameraMat(const Matrix4f& mat) {
	camera.setMat(mat);
}

/**
 * <!--  renderPre():  -->
 */
void Renderer::renderPre() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}
