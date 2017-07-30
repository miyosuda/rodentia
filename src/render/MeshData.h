// -*- C++ -*-
#ifndef MESHDATA_HEADER
#define MESHDATA_HEADER

#include <vector>
#include <string>
using namespace std;

class MeshFaceData;
class Material;
class Mesh;


class MeshData {
private:
	vector<MeshFaceData*> meshFaceDatas;
	vector<string> texturePathes;

public:
	~MeshData();
	void addMeshFace(MeshFaceData* meshFaceData, const string& texturePath);

	Mesh* toMesh(Material* material);
};

#endif
