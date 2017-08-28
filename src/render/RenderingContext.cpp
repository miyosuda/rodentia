#include "RenderingContext.h"

/**
 * <!--  initCamera():  -->
 */
void RenderingContext::initCamera(float ratio, bool flipping) {
	const float nearClip = 1.0f;
	const float farClip = 1000.0f;
	const float focalLength = 50.0f;
	
	camera.init(nearClip, farClip, focalLength, ratio, flipping);
}

/**
 * <!--  setCameraMat():  -->
 */
void RenderingContext::setCameraMat(const Matrix4f& mat) {
	camera.setMat(mat);
}
