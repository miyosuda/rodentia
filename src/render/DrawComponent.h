// -*- C++ -*-
#ifndef DRAWCOMPONENT_HEADER
#define DRAWCOMPONENT_HEADER

#include "Matrix4f.h"

class Vector3f;
class Mesh;
class RenderingContext;

class DrawComponent {
private:
	Matrix4f scaleMat;
	const Mesh* mesh;

public:
	DrawComponent(const Mesh* mesh_, const Vector3f& scale);
	~DrawComponent();
	void draw(RenderingContext& context, const Matrix4f& rigidBodyMat) const;
};

#endif
