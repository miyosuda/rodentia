#include "FrameBuffer.h"
#include <stdio.h>

FrameBuffer::FrameBuffer()
	:
	frameBufferId(0),
	frameBufferTextureId(0),
	depthRenderBufferId(0),
	width(0),
	height(0) {
}

FrameBuffer::~FrameBuffer() {
	release();
}

bool FrameBuffer::init(int width_, int height_) {
	width = width_;
	height = height_;
		
	release();
	
	glGenFramebuffers(1, &frameBufferId);
	glGenTextures(1, &frameBufferTextureId);

	bind();

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
				 width, height, 0, GL_RGBA,
				 GL_UNSIGNED_BYTE, 0);	
	
	// Set magnify & minify mode
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	
	// Set texture wrap mode
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	
	glBindFramebuffer(GL_FRAMEBUFFER, frameBufferId);

	// Set depth buffer
	glGenRenderbuffers(1, &depthRenderBufferId);
	glBindRenderbuffer(GL_RENDERBUFFER, depthRenderBufferId);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
							  GL_RENDERBUFFER, depthRenderBufferId);
	
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, frameBufferTextureId, 0);
	
	// Always check that our framebuffer is ok
	if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		printf("Failed to create frame buffer\n");
		return false;
	}
	
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
	
	return true;
}

void FrameBuffer::release() {
	if(frameBufferTextureId != 0) {
		glDeleteTextures(1, &frameBufferTextureId);
		frameBufferTextureId = 0;
	}
	if(frameBufferId != 0) {
		glDeleteFramebuffers(1, &frameBufferId);
		frameBufferId = 0;
	}
	if(depthRenderBufferId != 0) {
		glDeleteRenderbuffers(1, &depthRenderBufferId);
		depthRenderBufferId = 0;
	}
}

// レンダリング先としての利用を開始
// Start using as rendering target
void FrameBuffer::use() {
	glBindFramebuffer(GL_FRAMEBUFFER, frameBufferId);
}

// レンダリング先としての利用を終了
// End using as rendering target
void FrameBuffer::unuse() {
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// テクスチャとして利用を開始
// Start using as a texture
void FrameBuffer::bind() {
	glBindTexture(GL_TEXTURE_2D, frameBufferTextureId);
}

// テクスチャとして利用を終了
// End using as a texture
void FrameBuffer::unbind() {
	glBindTexture(GL_TEXTURE_2D, 0);
}

void FrameBuffer::setViewport() {
	glViewport(0, 0, width, height);
}
