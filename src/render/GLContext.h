// -*- C++ -*-
#ifndef GLCONTEXT_HEADER
#define GLCONTEXT_HEADER

#include <OpenGL/OpenGL.h>
#include <OpenGL/CGLTypes.h>
#include <OpenGL/CGLCurrent.h>

class GLContext {
private:
	CGLContextObj context;
	bool contextInitialized;

public:	
	GLContext();
	bool init();
	void release();
};

#endif
