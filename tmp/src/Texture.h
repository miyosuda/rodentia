// -*- C++ -*-
#ifndef TEXTURE_HEADER
#define TEXTURE_HEADER

class Texture {
public:
	int handle;
	Texture();
	~Texture();
	void release();
	void init(const unsigned char* data, int width, int height,
			  bool hasAlpha=true);
	void bind();
};

#endif
