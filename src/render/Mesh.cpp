#include "Mesh.h"

#include <float.h>

#include "MeshFace.h"
#include "Matrix4f.h"
#include "Vector3f.h"


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

void Mesh::calcBoundingBox(Vector3f& center, Vector3f& halfExtent) const {
	int size = meshFaces.size();

	Vector3f minPos(FLT_MAX, FLT_MAX, FLT_MAX);
	Vector3f maxPos(FLT_MIN, FLT_MIN, FLT_MIN);
	
	for(int i=0; i<size; ++i) {
		meshFaces[i]->calcBoundingBox(minPos, maxPos);
	}

	halfExtent.sub(maxPos, minPos);
	halfExtent *= 0.5f;

	center.add(maxPos, minPos);
	center *= 0.5f;
}
