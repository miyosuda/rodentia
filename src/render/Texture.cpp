#include "Texture.h"
#include "glinc.h"

Texture::Texture() 
    :
    handle(0)
{
}

Texture::~Texture() {
    release();
}

/**
 * <!--  release():  -->
 */
void Texture::release() {
    if( handle > 0 ) {
        glDeleteTextures(1, (unsigned int*)&handle);
        handle = 0;
    }
}

/**
 * <!--  init():  -->
 */
void Texture::init(const void* buffer, int width, int height,
                   bool hasAlpha, bool loop) {
    const unsigned char* data = (const unsigned char*)buffer;
    
    glGenTextures(1, (unsigned int*)&handle);
    bind();

    GLenum internalFormat;
    GLenum format;

    if( hasAlpha ) {
        internalFormat = GL_RGBA;
        format = GL_RGBA;
    } else {
        internalFormat = GL_RGB;
        format = GL_RGB;
    }

    GLenum type = GL_UNSIGNED_BYTE;

    glTexImage2D( GL_TEXTURE_2D, 0, internalFormat, width, height, 0,
                  format, 
                  type,
                  (const GLvoid*)data );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Use mipmap
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glGenerateMipmap(GL_TEXTURE_2D);
    
    if( loop ) {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    } else {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
}

/**
 * <!--  bind():  -->
 */
void Texture::bind() const {
    glBindTexture(GL_TEXTURE_2D, handle);
}
