// -*- C++ -*-
#ifndef RENDERER_HEADER
#define RENDERER_HEADER

#include "Matrix4f.h"
#include "Camera.h"

class Renderer {
private:
	void drawLine(const Vector4f& pos0, const Vector4f& pos1);
	
protected:
	Camera camera;
	int frameBufferWidth;
	int frameBufferHeight;
	bool flipping; // upside down flipping
	
	void drawFloor();

public:
	Renderer(bool flipping_=false)
		:
		frameBufferWidth(0),
		frameBufferHeight(0),
		flipping(flipping_) {
	}
	virtual ~Renderer() {
	}

	void setCameraMat(const Matrix4f& mat);
	const Camera& getCamera() const { return camera; }
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
