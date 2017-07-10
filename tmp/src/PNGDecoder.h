// -*- C++ -*-
#ifndef PNGDECODER_HEADER
#define PNGDECODER_HEADER

class Image;

class PNGDecoder {
public:
	static bool decode(unsigned char* buffer, int bufferSize, Image& image);
};

#endif
