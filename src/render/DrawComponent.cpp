#include "DrawComponent.h"
#include "RenderingContext.h"
#include "Mesh.h"
#include "Vector3f.h"

/**
 * <!--  DrawComponent():  -->
 */
DrawComponent::DrawComponent(Mesh* mesh_, const Vector3f& scale)
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

/**
 * <!--  calcBoundingBox():  -->
 */
void DrawComponent::calcBoundingBox(const Matrix4f& rigidBodyMat,
                                    BoundingBox& boundingBox) const {
    const BoundingBox& meshBoundingBox = mesh->getBoundingBox();
    float scaleX = scaleMat.getElement(0,0);
    float scaleY = scaleMat.getElement(1,1);
    float scaleZ = scaleMat.getElement(2,2);
    
    meshBoundingBox.transform(scaleX, scaleY, scaleZ,
                              rigidBodyMat,
                              boundingBox);
}

/**
 * <!--  replaceMaterials():  -->
 */
void DrawComponent::replaceMaterials(const vector<Material*>& materials) {
    mesh->replaceMaterials(materials);
}
