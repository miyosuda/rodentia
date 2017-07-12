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
	for (auto itr=textureMap.begin(); itr!=textureMap.end(); ++itr) {
		Texture* texture = itr->second;
		delete texture;
	}
	textureMap.clear();
}

/**
 * <!--  loadTexture():  -->
 */
Texture* TextureManager::loadTexture(const char* path) {
	auto itr = textureMap.find(path);
	if( itr != textureMap.end() ) {
		Texture* texture = textureMap[path];
		return texture;
	}
	
	int size;
	void* buffer = readFile(path, size);
	if( buffer == nullptr ) {
		return nullptr;
	}
	
	Image image;
	PNGDecoder::decode(buffer, size, image);

	Texture* texture = new Texture();
	texture->init(image.getBuffer(),
				  image.getWidth(), image.getHeight(),
				  image.hasAlpha());

	textureMap[path] = texture;
	return texture;	
}
