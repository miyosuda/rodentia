// -*- C++ -*-
#ifndef TEXTUREMANAGER_HEADER
#define TEXTUREMANAGER_HEADER

#include <map>
#include <string>

using namespace std;

class Texture;

class TextureManager {
private:
    map<string, Texture*> textureMap;
    void* readFile(const char* path, int& readSize);

    Texture* findTexture(const char* path);

public:
    ~TextureManager();
    void release();
    Texture* loadTexture(const char* path);
    Texture* getColorTexture(float r, float g, float b);
};

#endif
