// -*- C++ -*-
#ifndef RENDERER_HEADER
#define RENDERER_HEADER

#include "Matrix4f.h"

class Renderer {
private:
	void drawLine(const Vector4f& pos0, const Vector4f& pos1);
	
protected:
	Matrix4f camera;
	int frameBufferWidth;
	int frameBufferHeight;
	
	void setProjection(float width, float height);
	void drawFloor();	

public:
	Renderer()
		:
		frameBufferWidth(0),
		frameBufferHeight(0) {
	}
	virtual ~Renderer() {
	}

	void setCamera(const Matrix4f& mat);
	void renderPre();
	
	virtual bool init(int width, int height) = 0;
	virtual void render() = 0;	
	virtual void release() = 0;

	int getFrameBufferWidth()  const { return frameBufferWidth;  }
	int getFrameBufferHeight() const { return frameBufferHeight; }
	
	virtual const void* getBuffer() const {
		return nullptr;
	}
	int getFrameBufferSize() const {
		return frameBufferWidth * frameBufferHeight * 4;
	}
};

#endif
