// -*- C++ -*-
#ifndef OFFSCREENRENDERER_HEADER
#define OFFSCREENRENDERER_HEADER

#include "Renderer.h"
#include "glinc.h"
#include "GLContext.h"
#include "OffscreenFrameBuffer.h"


class OffscreenRenderer : public Renderer {
private:
	GLContext context;	
	void* buffer;
	OffscreenFrameBuffer frameBuffer;

public:	
	OffscreenRenderer()
		:
		Renderer(),
		buffer(nullptr) {
	}

	virtual bool init(int width, int height) override;
	virtual void release() override;
	virtual void renderPost() override;

	virtual const void* getBuffer() const override {
		return buffer;
	}
};


#endif
