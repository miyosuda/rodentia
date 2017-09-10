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
	setLightDir(Vector3f(-0.5, -1.0, -0.4));
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
	const float nearClip = 0.1f;
	const float farClip = 50.0f;
	const float focalLength = 50.0f;
	
	camera.initPerspective(nearClip, farClip, focalLength, ratio, flipping);
}

/**
 * <!--  setModelMat():  -->
 */
void RenderingContext::setModelMat(Matrix4f modelMat_) {
	modelMat.set(modelMat_);

	const Matrix4f& depthViewProjectionMat = lspsm.getLightViewProjection();
	depthModelViewProjectionMat.mul(depthViewProjectionMat, modelMat);

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
 * <!--  updateLSPSM():  -->
 */
void RenderingContext::updateLSPSM() {
	// TODO: 整理する
	
	const Matrix4f& mat = camera.getMat();
	const Vector4f& pos = mat.getColumnRef(3);
	const Vector4f& zaxis = mat.getColumnRef(2);

	Vector3f viewDir(-zaxis.x, -zaxis.y, -zaxis.z);

	lspsm.setNearClip(1.0f);
	
	lspsm.setViewDir(viewDir);
	lspsm.setLightDir(lightDir);

	lspsm.setEyeView(camera.getInvMat());
	lspsm.setEyePos(Vector3f(pos.x, pos.y, pos.z));
	lspsm.setEyeProjection(camera.getProjectionMat());
	lspsm.updateShadowMatrix();
}

/**
 * <!--  setCameraMat():  -->
 */
void RenderingContext::setCameraMat(const Matrix4f& mat) {
	camera.setMat(mat);
	
	updateLSPSM();
}

/**
 * <!--  setLightDir():  -->
 */
void RenderingContext::setLightDir(const Vector3f& lightDir_) {
	lightDir.set(lightDir_);
	lightDir.normalize();
}

/**
 * <!--  setBoundingBoxForShadow():  -->
 */
void RenderingContext::setBoundingBoxForShadow(const BoundingBox& boundingBox) {
	// TODO: リネームして、LiSPSMのBvolumeのclippingに使う.
}
