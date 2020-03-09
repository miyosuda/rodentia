// -*- C++ -*-
#ifndef RENDERTARGET_HEADER
#define RENDERTARGET_HEADER

#include "OffscreenFrameBuffer.h"
#include "DepthFrameBuffer.h"
#include "Vector3f.h"

class RenderTarget {
private:    
    int frameBufferWidth;
    int frameBufferHeight;
    Vector3f bgColor;

    void* buffer; // Frame buffer copy destination
    OffscreenFrameBuffer frameBuffer;
    DepthFrameBuffer depthFrameBuffer;

    int calcDepthFrameBufferSize(int width, int height);

public:    
    RenderTarget()
        :
        frameBufferWidth(0),
        frameBufferHeight(0),
        bgColor(0.0f, 0.0f, 0.0f),
        buffer(nullptr) {
    }

    bool init(int width, int height, const Vector3f& bgColor_,
              int shadowBufferWidth);

    void release();

    void prepareShadowDepthRendering();
    void prepareRendering();
    void finishRendering();

    int getFrameBufferWidth()  const { return frameBufferWidth;  }
    int getFrameBufferHeight() const { return frameBufferHeight; }

    const void* getBuffer() const {
        return buffer;
    }
    int getFrameBufferSize() const {
        return frameBufferWidth * frameBufferHeight * 3;
    }
};

#endif
