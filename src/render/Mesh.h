// -*- C++ -*-
#ifndef MESH_HEADER
#define MESH_HEADER

#include <vector>
using namespace std;

class MeshFace;
class Matrix4f;
class BoundingBox;
class RenderingContext;


class Mesh {
private:
	vector<MeshFace*> meshFaces;

public:
	Mesh() {}
	~Mesh();
	void addMeshFace(MeshFace* meshFace);
	void draw(const RenderingContext& context) const;
	void calcBoundingBox(BoundingBox& boundingBox) const;

};

#endif
