#include "MeshFace.h"
#include "Material.h"
#include "Matrix4f.h"
#include "MeshFace.h"

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
void MeshFace::draw( const Matrix4f& modelViewMat,
					 const Matrix4f& modelViewProjectionMat) {
	material->draw(meshFaceData,
				   modelViewMat,
				   modelViewProjectionMat);
}
