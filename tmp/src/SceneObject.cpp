#include "SceneObject.h"
#include "Camera.h"
#include "Mesh.h"
#include "Matrix4f.h"

/**
 * <!--  SceneObject():  -->
 */
SceneObject::SceneObject(const Mesh* mesh_)
	:
	mesh(mesh_) {
}

/**
 * <!--  draw():  -->
 */
void SceneObject::draw(const Camera& camera) {
	const Matrix4f& cameraInvMat = camera.getInvMat();
	const Matrix4f& projectionMat = camera.getProjectionMat();

	Matrix4f modelViewMat;
	modelViewMat.mul(cameraInvMat, mat);

	Matrix4f modelViewProjectionMat;
	modelViewProjectionMat.mul(projectionMat, modelViewMat);

	mesh->draw(modelViewMat,
			   modelViewProjectionMat);
}

/**
 * <!--  setMat():  -->
 */
void SceneObject::setMat(const Matrix4f& mat_) {
	mat.set(mat_);
}
