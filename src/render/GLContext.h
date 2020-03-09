// -*- C++ -*-
#ifndef GLCONTEXT_HEADER
#define GLCONTEXT_HEADER

#include "glad/glad.h" // for glad

#if defined(__APPLE__)

#include <OpenGL/OpenGL.h>
#include <OpenGL/CGLTypes.h>
#include <OpenGL/CGLCurrent.h>

class GLContext {
private:
    CGLContextObj context;
    bool contextInitialized;

public: 
    GLContext();
    bool init(int width, int height);
    void release();
};

#else // defined(__APPLE__)

#include <GL/glx.h>

class GLContext {
private:
    Display* display;
    GLXContext context;
    GLXPbuffer pbuffer;
    bool contextInitialized;

public: 
    GLContext();
    bool init(int width, int height);
    void release();
};


#endif // defined(__APPLE__)

#endif
