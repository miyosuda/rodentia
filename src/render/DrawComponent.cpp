#include "DrawComponent.h"
#include "Camera.h"
#include "Mesh.h"
#include "Vector3f.h"

/**
 * <!--  DrawComponent():  -->
 */
DrawComponent::DrawComponent(const Mesh* mesh_, const Vector3f& scale)
	:
	mesh(mesh_) {
	scaleMat.setIdentity();
	scaleMat.setElement(0, 0, scale.x);
	scaleMat.setElement(1, 1, scale.y);
	scaleMat.setElement(2, 2, scale.z);
}

/**
 * <!--  ~DrawComponent():  -->
 */
DrawComponent::~DrawComponent() {
	delete mesh;
}

/**
 * <!--  draw():  -->
 */
void DrawComponent::draw(const Camera& camera, const Matrix4f& rigidBodyMat) const {
	const Matrix4f& cameraInvMat = camera.getInvMat();
	const Matrix4f& projectionMat = camera.getProjectionMat();

	Matrix4f modelMat;
	modelMat.mul(rigidBodyMat, scaleMat);

	Matrix4f modelViewMat;
	modelViewMat.mul(cameraInvMat, modelMat);

	Matrix4f modelViewProjectionMat;
	modelViewProjectionMat.mul(projectionMat, modelViewMat);

	mesh->draw(modelMat,
			   modelViewMat,
			   modelViewProjectionMat);
}
