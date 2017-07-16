#include "Renderer.h"
#include "glinc.h"

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
