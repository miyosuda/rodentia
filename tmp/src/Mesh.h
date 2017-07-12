// -*- C++ -*-
#ifndef MESH_HEADER
#define MESH_HEADER

#include <vector>
using namespace std;

class MeshFace;

class Mesh {
private:
	vector<MeshFace*> meshFaces;

public:
	~Mesh() {
		int size = meshFaces.size();
		for(int i=0; i<size; ++i) {
			delete meshFaces[i];
		}
		meshFaces.clear();
	}
	
	void addMeshFace(MeshFace* meshFace) {
		meshFaces.push_back(meshFace);
	}

	void draw( const Matrix4f& modelViewMat, 
			   const Matrix4f& projectionMat ) {
		int size = meshFaces.size();
		for(int i=0; i<size; ++i) {
			meshFaces[i]->draw(modelViewMat, projectionMat);
		}
	}
};

#endif
