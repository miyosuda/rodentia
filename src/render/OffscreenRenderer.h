// -*- C++ -*-
#ifndef OFFSCREENRENDERER_HEADER
#define OFFSCREENRENDERER_HEADER

#include "Renderer.h"
#include "glinc.h"

class OffscreenRenderer : public Renderer {
private:
	GLFWwindow* window;
	void* buffer;

public:	
	OffscreenRenderer()
		:
		Renderer(),
		window(nullptr),
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
