// -*- C++ -*-
#ifndef MESH_HEADER
#define MESH_HEADER

#include <vector>
using namespace std;

class MeshFace;
class Matrix4f;

class Mesh {
private:
	vector<MeshFace*> meshFaces;

public:
	Mesh() {}
	~Mesh();
	void addMeshFace(MeshFace* meshFace);
	void draw( const Matrix4f& modelViewMat, 
			   const Matrix4f& projectionMat ) const;
};

#endif
