// -*- C++ -*-
#ifndef FRAMEBUFFER_HEADER
#define FRAMEBUFFER_HEADER

#include "glinc.h"

class FrameBuffer {
private:
	// id of frame buffer object
	GLuint frameBufferId;
	// id of color render buffer
	GLuint colorRenderBufferId;
	// id of depth render buffer
	GLuint depthRenderBufferId;
	
public:
	FrameBuffer();
	~FrameBuffer();
	bool init(int width_, int height_);
	void release();
	void use();
	void unuse();
};

#endif
