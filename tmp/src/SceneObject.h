// -*- C++ -*-
#ifndef SCENEOBJECT_HEADER
#define SCENEOBJECT_HEADER

#include "Matrix4f.h"

class Mesh;
class Camera;

class SceneObject {
private:
	Matrix4f mat;
	const Mesh* mesh;

public:
	SceneObject(const Mesh* mesh_);
	void draw(const Camera& camera);
	void setMat(const Matrix4f& mat_);
};

#endif
