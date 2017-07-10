#include "PNGDecoder.h"

#include <png.h> // libpng
#include <stdlib.h>
#include <memory.h>

#include "Image.h"


/**
 * <!--  pngReadFunc():  -->
 */
static void pngReadFunc( png_struct *pngobj,
						 png_bytep dstBuf, png_size_t size ) {
	unsigned char** p = (unsigned char**) png_get_io_ptr( pngobj );
	memcpy( dstBuf, *p, size );
	*p += (int)size;
}

/**
 * <!--  decode():  -->
 */
bool PNGDecoder::decode(unsigned char* buffer, int bufferSize, Image& image) {
	if( png_sig_cmp( buffer, 0, 8) ) {
		return false;
	}

	png_struct* pngobj = png_create_read_struct( PNG_LIBPNG_VER_STRING, NULL, NULL, NULL );

    if( pngobj == NULL ) {
		return false;
	}

	png_info* info = png_create_info_struct( pngobj );

    if( info == NULL ) {
		png_destroy_read_struct( &pngobj, NULL, NULL );
		return false;
	}

	if( setjmp(png_jmpbuf(pngobj)) ) {
		png_destroy_read_struct( &pngobj, NULL, NULL );
		return false;
	}

	unsigned char* filepos = buffer;
    png_set_read_fn( pngobj,
					 (png_voidp)&filepos,
					 (png_rw_ptr)pngReadFunc );

	png_read_info( pngobj, info );

	int width = png_get_image_width(pngobj, info);
	int height = png_get_image_height(pngobj, info);

	png_byte colorType = png_get_color_type(pngobj, info);
	//png_byte bitDepth = png_get_bit_depth(pngobj, info);
	//int channels = png_get_channels(pngobj, info);
	//int numberOfPasses = png_set_interlace_handling(pngobj);

	if (setjmp(png_jmpbuf(pngobj))) {
		return false;
	}

	bool hasAlpha = false;

	if( (colorType & PNG_COLOR_MASK_ALPHA) == 0 ) {
		// αが無い場合
		hasAlpha = false;
	} else {
		// αがある場合
		hasAlpha = true;
	}

	if( colorType & PNG_COLOR_TYPE_PALETTE ) {
		// index colorの場合
		png_set_palette_to_rgb(pngobj);

		if(png_get_valid( pngobj, info, PNG_INFO_tRNS ) ) {
			//png_set_tRNS_to_alpha(pngobj);
			hasAlpha = true;
        }
	}

	//png_bytep* lines = (png_bytep*) malloc(sizeof(png_bytep) * height);

	unsigned char **lines =
		(unsigned char **)malloc(sizeof(unsigned char *) * height);

	//printf("width=%d\n", width);
	//printf("height=%d\n", height);
	//printf("colorType=%d\n", colorType);
	//printf("bitDepth=%d\n", bitDepth);
	//printf("channels=%d\n", channels);
	//printf("pass=%d\n", numberOfPasses);

	if( hasAlpha ) {
		// to 32bit image
		image.init( width, height, Image::TYPE_32BIT );
	} else {
		// to 24bit image
		image.init( width, height, Image::TYPE_24BIT );
	}

	for( int i=0; i<height; i++ ) {
		lines[i] = (unsigned char*)image.getLineBuffer(i);
	}

	png_read_image( pngobj, lines );
	png_read_end( pngobj, NULL );

	free( lines );

	png_destroy_read_struct( &pngobj, &info, NULL );
	
	return true;
}
