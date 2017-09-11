// -*- C++ -*-
#ifndef RENDERER_HEADER
#define RENDERER_HEADER

#include "glinc.h"
#include "GLContext.h"
#include "OffscreenFrameBuffer.h"
#include "DepthFrameBuffer.h"
#include "Vector3f.h"


class Renderer {
private:
	int frameBufferWidth;
	int frameBufferHeight;
	Vector3f bgColor;

	GLContext context;
	void* buffer;
	OffscreenFrameBuffer frameBuffer;
	DepthFrameBuffer depthFrameBuffer;

	int calcDepthFrameBufferSize(int width, int height);

public:
	Renderer()
		:
		frameBufferWidth(0),
		frameBufferHeight(0),
		bgColor(0.0f, 0.0f, 0.0f),
		buffer(nullptr) {
	}

	bool init(int width, int height, const Vector3f& bgColor_);
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
