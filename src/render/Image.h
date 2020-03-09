// -*- C++ -*-
#ifndef IMAGE_HEADER
#define IMAGE_HEADER

class Image {
public:
    enum {
        TYPE_32BIT = 1,
        TYPE_24BIT = 2,
    };

private:
    int width;
    int height;
    int type;
    void* buffer;

    int getChannels() const;

public:
    Image();
    ~Image();
    void init(int width_, int height_, int type_);
    void release();
    void* getLineBuffer(int y);

    bool hasAlpha() const {
        return type == TYPE_32BIT;
    }
    int getWidth() const  {
        return width;
    }
    int getHeight() const {
        return height;
    }
    void* getBuffer() {
        return buffer;
    }
    const void* getBuffer() const {
        return buffer;
    }

    void debugDump();
};


#endif
