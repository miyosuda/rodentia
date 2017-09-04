// -*- C++ -*-
#ifndef RENDERER_HEADER
#define RENDERER_HEADER

#include "glinc.h"
#include "GLContext.h"
#include "OffscreenFrameBuffer.h"
#include "DepthFrameBuffer.h"

class Renderer {
private:
	int frameBufferWidth;
	int frameBufferHeight;

	GLContext context;
	void* buffer;
	OffscreenFrameBuffer frameBuffer;
	DepthFrameBuffer depthFrameBuffer;

public:
	Renderer()
		:
		frameBufferWidth(0),
		frameBufferHeight(0),
		buffer(nullptr) {
	}

	bool init(int width, int height);
	void prepareShadowDepthRendering();
	void prepareRendering();
	void finishRendering();
	void release();

	int getFrameBufferWidth()  const { return frameBufferWidth;  }
	int getFrameBufferHeight() const { return frameBufferHeight; }

	const void* getBuffer() const {
		return buffer;
	}	
	int getFrameBufferSize() const {
		return frameBufferWidth * frameBufferHeight * 3;
	}
};

#endif
