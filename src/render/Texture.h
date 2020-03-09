// -*- C++ -*-
#ifndef TEXTURE_HEADER
#define TEXTURE_HEADER

class Texture {
public:
    int handle;
    Texture();
    ~Texture();
    void release();
    void init(const void* buffer, int width, int height,
              bool hasAlpha=true,
              bool loop=true);
    void bind() const;
};

#endif
