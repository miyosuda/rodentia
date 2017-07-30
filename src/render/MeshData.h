// -*- C++ -*-
#ifndef MESHDATA_HEADER
#define MESHDATA_HEADER

#include <vector>
#include <string>
using namespace std;

class MeshFaceData;

class MeshData {
private:
	vector<MeshFaceData*> meshFaceDatas;
	vector<string> texturePathes;

public:
	void addMeshFace(MeshFaceData* meshFaceData, const string& texturePath);
};

#endif
