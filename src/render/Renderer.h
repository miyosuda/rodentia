// -*- C++ -*-
#ifndef RENDERER_HEADER
#define RENDERER_HEADER

#include "Matrix4f.h"
#include "Camera.h"

class Renderer {
protected:
	Camera camera;
	int frameBufferWidth;
	int frameBufferHeight;

	void initCamera(float ratio, bool flipping);
	
public:
	Renderer()
		:
		frameBufferWidth(0),
		frameBufferHeight(0) {
	}
	virtual ~Renderer() {
	}

	void setCameraMat(const Matrix4f& mat);
	const Camera& getCamera() const { return camera; }
	void renderPre();
	
	virtual bool init(int width, int height) = 0;
	virtual void renderPost() = 0;	
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
