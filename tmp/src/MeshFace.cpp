#include "MeshFace.h"
#include "Material.h"
#include "Matrix4f.h"

/**
 * <!--  MeshFace():  -->
 */
MeshFace::MeshFace( Material* material_,
					float* vertices_,
					int verticesSize_,
					short* indices_,
					int indicesSize_ )
	:
	material(material_),
	vertices(vertices_),
	verticesSize(verticesSize_),
	indices(indices_),
	indicesSize(indicesSize_) {
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
	material->draw(vertices,
				   indices,
				   indicesSize,
				   modelViewMat,
				   modelViewProjectionMat);
}
