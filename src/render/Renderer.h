// -*- C++ -*-
#ifndef RENDERER_HEADER
#define RENDERER_HEADER

#include "glinc.h"
#include "GLContext.h"
#include "RenderTarget.h"


class Renderer {
private:
    RenderTarget renderTarget;
    GLContext context;

public:
    Renderer();
    bool init(int width, int height, const Vector3f& bgColor_);
    void prepareShadowDepthRendering();
    void prepareRendering();
    void finishRendering();
    void release();

    int getFrameBufferWidth()  const { return renderTarget.getFrameBufferWidth();  }
    int getFrameBufferHeight() const { return renderTarget.getFrameBufferHeight(); }

    const void* getBuffer() const {
        return renderTarget.getBuffer();
    }
    int getFrameBufferSize() const {
        return renderTarget.getFrameBufferSize();
    }
};

#endif
