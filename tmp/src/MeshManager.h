// -*- C++ -*-
#ifndef MESHMANAGER_HEADER
#define MESHMANAGER_HEADER

class MeshFaceData;
class Mesh;
class Texture;
class Shader;

class MeshManager {
private:
	MeshFaceData* boxMeshFaceData;

public:
	~MeshManager();
	const Mesh* getBoxMesh(Texture* texture, Shader* shader);
};

#endif
