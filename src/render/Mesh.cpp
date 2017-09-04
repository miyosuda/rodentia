#include "Mesh.h"

#include <float.h>

#include "MeshFace.h"
#include "Matrix4f.h"
#include "BoundingBox.h"


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

void Mesh::draw(const RenderingContext& context) const {
	int size = meshFaces.size();
	for(int i=0; i<size; ++i) {
		meshFaces[i]->draw(context);
	}
}

void Mesh::calcBoundingBox(BoundingBox& boundingBox) const {
	int size = meshFaces.size();
	for(int i=0; i<size; ++i) {
		boundingBox.merge(meshFaces[i]->getBoundingBox());
	}
}
