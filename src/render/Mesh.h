// -*- C++ -*-
#ifndef MESH_HEADER
#define MESH_HEADER

#include <vector>
using namespace std;

class MeshFace;
class Matrix4f;
class Vector3f;
class RenderingContext;


class Mesh {
private:
	vector<MeshFace*> meshFaces;

public:
	Mesh() {}
	~Mesh();
	void addMeshFace(MeshFace* meshFace);
	void draw(const RenderingContext& context) const;
	void calcBoundingBox(Vector3f& center, Vector3f& halfExtent) const;

	int debugGetMeshFaceSize() const {
		return meshFaces.size();
	}
	const MeshFace* debugGetMeshFace(int index) const {
		return meshFaces[index];
	}
};

#endif
