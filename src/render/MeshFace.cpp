#include "MeshFace.h"
#include "Material.h"
#include "Matrix4f.h"
#include "MeshFace.h"
#include "MeshFaceData.h"

/**
 * <!--  MeshFace():  -->
 */
MeshFace::MeshFace( Material* material_,
                    const MeshFaceData& meshFaceData_ )
    :
    material(material_),
    meshFaceData(meshFaceData_) {
}

/**
 * <!--  ~MeshFace():  -->
 */
MeshFace::~MeshFace() {
    delete material;
}

/**
 * <!--  draw():  -->
 */
void MeshFace::draw(const RenderingContext& context) {
    material->draw(meshFaceData, context);
}

/**
 * <!--  getBoundingBox():  -->
 */
const BoundingBox& MeshFace::getBoundingBox() const {
    return meshFaceData.getBoundingBox();
}

/**
 * <!--  replaceMaterial():  -->
 */
void MeshFace::replaceMaterial(Material* material_) {
    delete material;
    material = material_;
}
