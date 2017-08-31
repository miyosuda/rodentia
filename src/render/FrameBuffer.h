// -*- C++ -*-
#ifndef FRAMEBUFFER_HEADER
#define FRAMEBUFFER_HEADER

#include "glinc.h"

class FrameBuffer {
private:
	// id of frame buffer object
	GLuint frameBufferId;
	// id of texture
	GLuint frameBufferTextureId;
	
	int width;
	int height;
	
public:
	FrameBuffer();
	~FrameBuffer();
	bool init(int width_, int height_);
	bool initForDepth(int width_, int height_);
	void release();
	void use();
	void unuse();
	void bind();
	void unbind();
	void setViewport();
};

#endif
