// -*- C++ -*-
#ifndef OFFSCREENFRAMEBUFFER_HEADER
#define OFFSCREENFRAMEBUFFER_HEADER

#include "glinc.h"

class OffscreenFrameBuffer {
private:
    // id of frame buffer object
    GLuint frameBufferId;
    // id of color render buffer
    GLuint colorRenderBufferId;
    // id of depth render buffer
    GLuint depthRenderBufferId;

    int width;
    int height;
    
public:
    OffscreenFrameBuffer();
    ~OffscreenFrameBuffer();
    bool init(int width_, int height_);
    void release();
    void use();
    void unuse();
    void setViewport();
};

#endif
