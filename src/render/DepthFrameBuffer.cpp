#include "DepthFrameBuffer.h"
#include <stdio.h>

DepthFrameBuffer::DepthFrameBuffer()
    :
    frameBufferId(0),
    frameBufferTextureId(0),
    width(0),
    height(0) {
}

DepthFrameBuffer::~DepthFrameBuffer() {
    release();
}
    
bool DepthFrameBuffer::init(int width_, int height_) {
    width = width_;
    height = height_;
        
    release();
    
    glGenFramebuffers(1, &frameBufferId);
    glGenTextures(1, &frameBufferTextureId);

    bind();
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT16,
                 width, height, 0, GL_DEPTH_COMPONENT,
                 GL_FLOAT,
                 0);
    
    // Set magnify & minify mode
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    
    // Set texture wrap mode
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    // もしＲの値がテクスチャの値以下なら真（つまり日向）
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    
    // 書き込むポリゴンのテクスチャ座標値のＲとテクスチャとの比較を行うようにする
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
    
    // 比較の結果を輝度値として得る
    // Get comparing result as a luminance value
    //glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE);
    
    glBindFramebuffer(GL_FRAMEBUFFER, frameBufferId);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, frameBufferTextureId, 0);
    
    // No color output in the bound framebuffer, only depth.
    glDrawBuffer(GL_NONE);
        
    // Always check that our framebuffer is ok
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        printf("Failed to create frame buffer\n");
        return false;
    }
        
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
        
    return true;
}

void DepthFrameBuffer::release() {
    if (frameBufferTextureId != 0) {
        glDeleteTextures(1, &frameBufferTextureId);
        frameBufferTextureId = 0;
    }
    if (frameBufferId != 0) {
        glDeleteFramebuffers(1, &frameBufferId);
        frameBufferId = 0;
    }
}

// Start using as rendering target
void DepthFrameBuffer::use() {
    glBindFramebuffer(GL_FRAMEBUFFER, frameBufferId);
}

// End using as rendering target
void DepthFrameBuffer::unuse() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// Start using as a texture
void DepthFrameBuffer::bind() {
    glBindTexture(GL_TEXTURE_2D, frameBufferTextureId);
}

// End using as a texture
void DepthFrameBuffer::unbind() {
    glBindTexture(GL_TEXTURE_2D, 0);
}
    
void DepthFrameBuffer::setViewport() {
    glViewport(0, 0, width, height);
}
