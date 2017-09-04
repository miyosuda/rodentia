#include "RenderingContext.h"
#include "BoundingBox.h"


/**
 * <!--  RenderingContext():  -->
 */
RenderingContext::RenderingContext()
	:
	depthBiasMat(
		0.5f, 0.0f, 0.0f, 0.0f,
		0.0f, 0.5f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.5f, 0.0f,
		0.5f, 0.5f, 0.5f, 1.0f)
{
	setPath(SHADOW);
	setLightDir(Vector3f(1.0f, -0.4f, 0.3f));
}

/**
 * <!--  setPath():  -->
 */
void RenderingContext::setPath(Path path_) {
	path = path_;
}

/**
 * <!--  initCamera():  -->
 */
void RenderingContext::initCamera(float ratio, bool flipping) {
	const float nearClip = 1.0f;
	const float farClip = 1000.0f;
	const float focalLength = 50.0f;
	
	camera.initPerspective(nearClip, farClip, focalLength, ratio, flipping);

	// TODO: 状況におうじて幅を変える必要あり
	//lightCamera.initOrtho(-10.0f, 20.0f, 20.0f, 20.0f);
	lightCamera.initOrtho(-10.0f, 20.0f, -20.0f, 20.0f, -20.0f, 20.0f);
}

/**
 * <!--  setModelMat():  -->
 */
void RenderingContext::setModelMat(Matrix4f modelMat_) {
	modelMat.set(modelMat_);

	// Set matrix for shadow depth rendering & normal rendering
	const Matrix4f& depthViewMat = lightCamera.getInvMat();
	const Matrix4f& depthProjectionMat = lightCamera.getProjectionMat();

	Matrix4f depthModelViewMat;
	depthModelViewMat.mul(depthViewMat, modelMat);
	// Used both for shadow depth rendering and normal rendering
	depthModelViewProjectionMat.mul(depthProjectionMat, depthModelViewMat);

	if( !isRenderingShadow() ) {
		// Set matrix for normal rendering
		const Matrix4f& viewMat = camera.getInvMat();
		const Matrix4f& projectionMat = camera.getProjectionMat();

		modelViewMat.mul(viewMat, modelMat);
		modelViewProjectionMat.mul(projectionMat, modelViewMat);

		depthBiasModelViewProjectionMat.mul(depthBiasMat, depthModelViewProjectionMat);
	}
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
	
	lightCamera.lookAt( Vector3f(0.0f, 0.0f, 0.0f),
						lightDir,
						Vector3f(0.0f, 1.0f, 0.0f) );
}

/**
 * <!--  setBoundingBoxForShadow():  -->
 */
void RenderingContext::setBoundingBoxForShadow(const BoundingBox& boundingBox) {
	const Matrix4f& mat = lightCamera.getInvMat();

	BoundingBox transformedBoundingBox;
	boundingBox.transform(1.0f, 1.0f, 1.0f,
						  mat,
						  transformedBoundingBox);
	
	const Vector3f& minPos = transformedBoundingBox.getMinPos();
	const Vector3f& maxPos = transformedBoundingBox.getMaxPos();

	lightCamera.initOrtho(minPos.z, maxPos.z,
						  minPos.x, maxPos.x,
						  minPos.y, maxPos.y);
}
