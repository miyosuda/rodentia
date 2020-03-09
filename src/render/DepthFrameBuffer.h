// -*- C++ -*-
#ifndef DEPTHFRAMEBUFFER_HEADER
#define DEPTHFRAMEBUFFER_HEADER

#include "glinc.h"

class DepthFrameBuffer {
private:
    // id of frame buffer object
    GLuint frameBufferId;
    // id of texture
    GLuint frameBufferTextureId;
    
    int width;
    int height;
    
public:
    DepthFrameBuffer();
    ~DepthFrameBuffer();
    bool init(int width_, int height_);
    void release();
    void use();
    void unuse();
    void bind();
    void unbind();
    void setViewport();
};

#endif
