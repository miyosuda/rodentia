// -*- C++ -*-
#ifndef BUFFEROBJECTS_HEADER
#define BUFFEROBJECTS_HEADER

#include "glinc.h"

class VertexArray {
private:
    GLuint handle;

public:
    VertexArray()
        :
        handle(0) {
    }
    
    ~VertexArray() {
        release();
    }

    void release() {
        if( handle > 0 ) {
            glDeleteVertexArrays(1, &handle);
            handle = 0;
        }
    }
    
    bool init() {
        glGenVertexArrays(1, &handle);
        bind();
        return glGetError() == GL_NO_ERROR;
    }

    void bind() const {
        glBindVertexArray(handle);
    }

    void unbind() const {
        glBindVertexArray(0);
    }
};


class VertexBuffer {
private:
    GLuint handle;

public:
    VertexBuffer()
        :
        handle(0) {
    }
    
    ~VertexBuffer() {
        release();
    }

    void release() {
        if( handle > 0 ) {
            glDeleteBuffers(1, &handle);
            handle = 0;
        }
    }
    
    bool init(const float* array, int arraySize, bool dynamic=false) {
        glGenBuffers(1, &handle);
        bind();

        if( dynamic ) {
            glBufferData(GL_ARRAY_BUFFER, sizeof(float) * arraySize, array, GL_DYNAMIC_DRAW);
        } else {
            glBufferData(GL_ARRAY_BUFFER, sizeof(float) * arraySize, array, GL_STATIC_DRAW);
        }
        
        return glGetError() == GL_NO_ERROR;
    }

    // Used only for debug purpose. (LineShader for debug drawing)
    void modify(const float* array, int arraySize) {
        bind();
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * arraySize, array);
    }

    void bind() const {
        glBindBuffer(GL_ARRAY_BUFFER, handle);
    }

    void unbind() const {
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
};

class IndexBuffer {
private:
    GLuint handle;

public:
    IndexBuffer()
        :
        handle(0) {
    }
    
    ~IndexBuffer() {
        release();
    }

    void release() {
        if( handle > 0 ) {
            glDeleteBuffers(1, &handle);
            handle = 0;
        }
    }
    
    bool init(const unsigned short* array, int arraySize) {
        glGenBuffers(1, &handle);
        bind();
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned short) * arraySize,
                     array, GL_STATIC_DRAW);
        return glGetError() == GL_NO_ERROR;
    }

    void bind() const{
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, handle);
    }

    void unbind() const {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }
};

#endif
