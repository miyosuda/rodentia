// -*- C++ -*-
#ifndef RENDERER_HEADER
#define RENDERER_HEADER

class Renderer {
protected:
	int frameBufferWidth;
	int frameBufferHeight;

public:
	Renderer()
		:
		frameBufferWidth(0),
		frameBufferHeight(0) {
	}
	virtual ~Renderer() {
	}

	void renderPre();
	
	virtual bool init(int width, int height) = 0;
	virtual void renderPost() = 0;	
	virtual void release() = 0;

	int getFrameBufferWidth()  const { return frameBufferWidth;  }
	int getFrameBufferHeight() const { return frameBufferHeight; }
	
	virtual const void* getBuffer() const {
		return nullptr;
	}
	int getFrameBufferSize() const {
		return frameBufferWidth * frameBufferHeight * 3;
	}
};

#endif
