#include "TextureManager.h"

#include <stdlib.h>

#include "Texture.h"
#include "PNGDecoder.h"
#include "Image.h"

/**
 * <!--  readFile():  -->
 */
void* TextureManager::readFile(const char* path, int& readSize) {
    readSize = 0;
    
    FILE* file = fopen(path, "rb");
    if ( file == nullptr ) {
        printf("Couldn't open file: %s\n", path);
        return nullptr;
    }

    int pos = ftell(file);
    fseek(file, 0, SEEK_END);
    
    int size = ftell(file);
    fseek(file, pos, SEEK_SET);

    void* buffer = malloc(size);
    int ret = fread(buffer, 1, size, file);
    if( ret != size ) {
        fclose(file);
        free(buffer);
        printf("Failed to read file: %s\n", path);
        return nullptr;
    }

    readSize = size;
    
    fclose(file);
    return buffer;
}

/**
 * <!--  ~TextureManager():  -->
 */
TextureManager::~TextureManager() {
    release();
}

/**
 * <!--  release():  -->
 */
void TextureManager::release() {
    for (auto itr=textureMap.begin(); itr!=textureMap.end(); ++itr) {
        Texture* texture = itr->second;
        delete texture;
    }
    textureMap.clear();
}

/**
 * <!--  findTexture():  -->
 */
Texture* TextureManager::findTexture(const char* path) {
    auto itr = textureMap.find(path);
    if( itr != textureMap.end() ) {
        Texture* texture = itr->second;
        return texture;
    } else {
        return nullptr;
    }
}

/**
 * <!--  loadTexture():  -->
 */
Texture* TextureManager::loadTexture(const char* path) {
    Texture* texture = findTexture(path);
    if( texture != nullptr ) {
        return texture;
    }
    
    int size;
    void* buffer = readFile(path, size);
    if( buffer == nullptr ) {
        return nullptr;
    }
    
    Image image;
    PNGDecoder::decode(buffer, size, image);

    texture = new Texture();
    texture->init(image.getBuffer(),
                  image.getWidth(), image.getHeight(),
                  image.hasAlpha());

    textureMap[path] = texture;
    return texture; 
}

/**
 * <!--  getColorTexture():  -->
 */
Texture* TextureManager::getColorTexture(float r, float g, float b) {
    int ir = (int)(255.0f * r);
    int ig = (int)(255.0f * g);
    int ib = (int)(255.0f * b);

    char nameBuf[32];
    sprintf(nameBuf, "__%02x%02x%02x", ir, ig, ib);

    Texture* texture = findTexture(nameBuf);
    if( texture != nullptr ) {
        return texture;
    }

    const int w = 8;
    const int h = 8;

    Image image;
    image.init(w, h, Image::TYPE_24BIT);
    
    unsigned char* buffer = (unsigned char*)image.getBuffer();
    for(int i=0; i<w*h; ++i) {
        *buffer++ = (unsigned char)ir;
        *buffer++ = (unsigned char)ig;
        *buffer++ = (unsigned char)ib;
    }
    
    texture = new Texture();
    texture->init(image.getBuffer(),
                  image.getWidth(), image.getHeight(),
                  image.hasAlpha());
    
    textureMap[nameBuf] = texture;
    return texture;
}
