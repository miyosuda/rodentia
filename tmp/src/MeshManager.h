// -*- C++ -*-
#ifndef MESHMANAGER_HEADER
#define MESHMANAGER_HEADER

class MeshFaceData;
class Mesh;
class Material;

class MeshManager {
private:
	MeshFaceData* boxMeshFaceData;
	MeshFaceData* sphereMeshFaceData;

public:
	MeshManager()
		:
		boxMeshFaceData(nullptr),
		sphereMeshFaceData(nullptr) {
	}
	~MeshManager();
	const Mesh* getBoxMesh(Material* material);
	const Mesh* getSphereMesh(Material* material);
};

#endif
