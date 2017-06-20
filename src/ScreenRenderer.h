// -*- C++ -*-
#ifndef SCREENRENDERER_HEADER
#define SCREENRENDERER_HEADER

#include "Renderer.h"
#include <GLFW/glfw3.h>

class ScreenRenderer : public Renderer {
private:

public:	
	void init(int width, int height);
	virtual void render() override;
};

#endif
