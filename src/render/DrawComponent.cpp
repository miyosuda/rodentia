#include "DrawComponent.h"
#include "RenderingContext.h"
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
void DrawComponent::draw(RenderingContext& context, const Matrix4f& rigidBodyMat) const {
	Matrix4f modelMat;
	modelMat.mul(rigidBodyMat, scaleMat);

	context.setModelMat(modelMat);
	
	mesh->draw(context);
}
