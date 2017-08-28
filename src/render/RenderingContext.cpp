#include "RenderingContext.h"


/**
 * <!--  RenderingContext():  -->
 */
RenderingContext::RenderingContext() {
	setLightDir(Vector3f(1.0f, -0.4f, 0.3f));
}

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
 * <!--  setModelMat():  -->
 */
void RenderingContext::setModelMat(Matrix4f modelMat_) {
	modelMat.set(modelMat_);

	const Matrix4f& viewMat = camera.getInvMat();
	const Matrix4f& projectionMat = camera.getProjectionMat();

	modelViewMat.mul(viewMat, modelMat);
	modelViewProjectionMat.mul(projectionMat, modelViewMat);
}

/**
 * <!--  setCameraMat():  -->
 */
void RenderingContext::setCameraMat(const Matrix4f& mat) {
	camera.setMat(mat);
}

/**
 * <!--  setLightDir():  -->
 */
void RenderingContext::setLightDir(const Vector3f& lightDir_) {
	lightDir.set(lightDir_);
	lightDir.normalize();
}
