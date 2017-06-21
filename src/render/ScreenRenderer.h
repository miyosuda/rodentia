// -*- C++ -*-
#ifndef SCREENRENDERER_HEADER
#define SCREENRENDERER_HEADER

#include "Renderer.h"
#include <GLFW/glfw3.h>

class ScreenRenderer : public Renderer {
private:

public:
	ScreenRenderer()
		:
		Renderer() {
	}
	virtual bool init(int width, int height) override;
	virtual void render() override;
	virtual void release() override;
};

#endif
