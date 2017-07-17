#include "Mesh.h"
#include "MeshFace.h"
#include "Matrix4f.h"

Mesh::~Mesh() {
	int size = meshFaces.size();
	for(int i=0; i<size; ++i) {
		delete meshFaces[i];
	}
	meshFaces.clear();
}
	
void Mesh::addMeshFace(MeshFace* meshFace) {
	meshFaces.push_back(meshFace);
}

void Mesh::draw( const Matrix4f& modelMat,
				 const Matrix4f& modelViewMat, 
				 const Matrix4f& projectionMat ) const {
	int size = meshFaces.size();
	for(int i=0; i<size; ++i) {
		meshFaces[i]->draw(modelMat, modelViewMat, projectionMat);
	}
}
