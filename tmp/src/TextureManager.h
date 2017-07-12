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

public:
	~TextureManager();
	Texture* loadTexture(const char* path);
};

#endif
