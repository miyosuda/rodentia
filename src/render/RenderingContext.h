// -*- C++ -*-
#ifndef RENDERINGCONTEXT_HEADER
#define RENDERINGCONTEXT_HEADER

#include "Matrix4f.h"
#include "Camera.h"

class RenderingContext {
private:
	Camera camera;

public:
	void initCamera(float ratio, bool flipping);
	
	void setCameraMat(const Matrix4f& mat);
	const Camera& getCamera() const { return camera; }
};

#endif
