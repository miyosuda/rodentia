// -*- C++ -*-
#ifndef MESHMANAGER_HEADER
#define MESHMANAGER_HEADER

class MeshFaceData;
class Mesh;
class Material;

class MeshManager {
private:
	MeshFaceData* boxMeshFaceData;

public:
	MeshManager()
		:
		boxMeshFaceData(nullptr) {
	}
	~MeshManager();
	const Mesh* getBoxMesh(Material* material);
};

#endif
