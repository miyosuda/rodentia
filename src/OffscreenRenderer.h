// -*- C++ -*-
#ifndef OFFSCREENRENDERER_HEADER
#define OFFSCREENRENDERER_HEADER

#include "Renderer.h"
#include <GLFW/glfw3.h>

class OffscreenRenderer : public Renderer {
private:
	GLFWwindow* window;
	int frameBufferWidth;
	int frameBufferHeight;	
	void* buffer;

public:	
	OffscreenRenderer()
		:
		window(nullptr),
		frameBufferWidth(0),
		frameBufferHeight(0),
		buffer(nullptr) {
	}

	bool init(int width, int height);
	void release();
	virtual void render() override;

	int getFrameBufferWidth()  const { return frameBufferWidth;  }
	int getFrameBufferHeight() const { return frameBufferHeight; }
	const void* getBuffer()    const { return buffer; }

	int getFrameBufferSize() const {
		return frameBufferWidth * frameBufferHeight * 4;
	}
};


#endif
