#include "MeshData.h"

void MeshData::addMeshFace(MeshFaceData* meshFaceData, const string& texturePath) {
	meshFaceDatas.push_back(meshFaceData);
	texturePathes.push_back(texturePath);
}
