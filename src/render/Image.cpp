#include "Image.h"
#include <stdio.h>
#include <stdlib.h> // nullptr

/**
 * <!--  Image():  -->
 */
Image::Image() 
    :
    buffer(nullptr)
{
    width = 0;
    height = 0;
    type = 0;
}

/**
 * <!--  ~Image():  -->
 */
Image::~Image() {
    release();
}

/**
 * <!--  init():  -->
 */
void Image::init(int width_, int height_, int type_) {
    width = width_;
    height = height_;
    type = type_;

    // buffer order is R,G,B,(A), R,G,B,(A) ....
    int b = getChannels();
    int bufferSize = b * width * height;
    buffer = malloc(bufferSize);
}

/**
 * <!--  release():  -->
 */
void Image::release() {
    if( buffer != nullptr ) {
        free(buffer);
        buffer = nullptr;
    }
}

/**
 * <!--  getChannels():  -->
 */
int Image::getChannels() const {
    int b;
    if( type == TYPE_32BIT ) {
        b = 4;
    } else {
        b = 3;
    }
    return b;
}

/**
 * <!--  getLineBuffer():  -->
 */
void* Image::getLineBuffer(int y) {
    unsigned char *buf = (unsigned char*)buffer;

    int b = getChannels();
    buf += (width * b * y);
    return buf;
}

/**
 * <!--  debugDump():  -->
 */
void Image::debugDump() {
    unsigned char* buf = (unsigned char*)buffer;

    int pos = 0;
    int channels = getChannels();
    for(int j=0; j<height; ++j) {
        for(int i=0; i<width; ++i) {
            for(int c=0; c<channels; ++c) {
                unsigned char val = buf[pos];
                pos++;
                printf("[%d,%d,%d]=%d\n", j,i,c, val);
            }
        }
    }
}
