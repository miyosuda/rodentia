// -*- C++ -*-
#ifndef RENDERER_HEADER
#define RENDERER_HEADER

#include "Matrix4f.h"

class Renderer {
private:
	void drawLine(const Vector4f& pos0, const Vector4f& pos1);
	
protected:
	Matrix4f camera;
	
	void setProjection(float width, float height);
	void drawFloor();	

public:
	void setCamera(const Matrix4f& mat);
	void renderPre();
	virtual void render() = 0;
};

#endif
