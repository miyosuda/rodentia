#include "MeshData.h"
#include "Mesh.h"
#include "MeshFace.h"
#include "MeshFaceData.h"

MeshData::~MeshData() {
	for(int i=0; i<meshFaceDatas.size(); ++i) {
		delete meshFaceDatas[i];
	}
	meshFaceDatas.clear();
}

void MeshData::addMeshFace(MeshFaceData* meshFaceData, const string& texturePath) {
	meshFaceDatas.push_back(meshFaceData);
	texturePathes.push_back(texturePath);
}

Mesh* MeshData::toMesh(Material* material) {
	Mesh* mesh = new Mesh();

	for(int i=0; i<meshFaceDatas.size(); ++i) {
		MeshFace* meshFace = new MeshFace(material,
										  *meshFaceDatas[i]);
		mesh->addMeshFace(meshFace);
	}

	return mesh;
}
