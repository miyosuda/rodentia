#include "OffscreenFrameBuffer.h"
#include <stdio.h>


/**
 * <!--  OffscreenFrameBuffer():  -->
 */
OffscreenFrameBuffer::OffscreenFrameBuffer()
    :
    frameBufferId(0),
    colorRenderBufferId(0),
    depthRenderBufferId(0),
    width(0),
    height(0) {
}

/**
 * <!--  ~OffscreenFrameBuffer():  -->
 */
OffscreenFrameBuffer::~OffscreenFrameBuffer() {
    release();
}

/**
 * <!--  init():  -->
 */
bool OffscreenFrameBuffer::init(int width_, int height_) {
    width = width_;
    height = height_;
    
    release();

    // Color render buffer
    glGenRenderbuffers(1, &colorRenderBufferId);
    glBindRenderbuffer(GL_RENDERBUFFER, colorRenderBufferId);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB, width, height);

    // Set depth buffer
    glGenRenderbuffers(1, &depthRenderBufferId);
    glBindRenderbuffer(GL_RENDERBUFFER, depthRenderBufferId);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);

    // Frame buffer
    glGenFramebuffers(1, &frameBufferId);   
    glBindFramebuffer(GL_FRAMEBUFFER, frameBufferId);

    // Attach color render buffer to frame buffer
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER,
                              colorRenderBufferId);

    // Attach depth render buffer to frame buffer
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER,
                              depthRenderBufferId);
    
    // Always check that our framebuffer is ok
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        printf("Failed to create frame buffer\n");
        return false;
    }
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    
    return true;
}

/**
 * <!--  release():  -->
 */
void OffscreenFrameBuffer::release() {
    if(frameBufferId != 0) {
        glDeleteFramebuffers(1, &frameBufferId);
        frameBufferId = 0;
    }
    if(colorRenderBufferId != 0) {
        glDeleteRenderbuffers(1, &colorRenderBufferId);
        colorRenderBufferId = 0;
    }
    if(depthRenderBufferId != 0) {
        glDeleteRenderbuffers(1, &depthRenderBufferId);
        depthRenderBufferId = 0;
    }
}

// Start using as rendering target
void OffscreenFrameBuffer::use() {
    glBindFramebuffer(GL_FRAMEBUFFER, frameBufferId);
}

// End using as rendering target
void OffscreenFrameBuffer::unuse() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void OffscreenFrameBuffer::setViewport() {
    glViewport(0, 0, width, height);
}
