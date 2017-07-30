// -*- C++ -*-
#ifndef MESHMANAGER_HEADER
#define MESHMANAGER_HEADER

class Mesh;
class MeshData;
class MeshFaceData;
class Material;


class MeshManager {
private:
	MeshData* boxMeshData;
	MeshData* sphereMeshData;
	// TODO: map<string, MeshData*> modelMeshDataMap;

public:
	MeshManager()
		:
		boxMeshData(nullptr),
		sphereMeshData(nullptr) {
	}
	~MeshManager();
	const Mesh* getBoxMesh(Material* material);
	const Mesh* getSphereMesh(Material* material);
	const Mesh* getModelMesh(const char* path);
};

#endif
