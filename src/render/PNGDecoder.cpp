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
bool PNGDecoder::decode(void* buffer, int bufferSize, Image& image) {
    unsigned char* buf = (unsigned char*)buffer;
    
    if( png_sig_cmp(buf, 0, 8) ) {
        return false;
    }

    png_struct* pngobj = png_create_read_struct( PNG_LIBPNG_VER_STRING, NULL, NULL, NULL );

    if( pngobj == NULL ) {
        printf("failed to setup libpng\n");
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

    unsigned char* filepos = buf;
    png_set_read_fn( pngobj,
                     (png_voidp)&filepos,
                     (png_rw_ptr)pngReadFunc );

    png_read_info( pngobj, info );

    int width = png_get_image_width(pngobj, info);
    int height = png_get_image_height(pngobj, info);

    png_byte colorType = png_get_color_type(pngobj, info);

    if (setjmp(png_jmpbuf(pngobj))) {
        return false;
    }

    bool hasAlpha = false;

    if( (colorType & PNG_COLOR_MASK_ALPHA) == 0 ) {
        // with no alpha
        hasAlpha = false;
    } else {
        // with alpha
        hasAlpha = true;
    }

    if( colorType & PNG_COLOR_TYPE_PALETTE ) {
        // index color
        png_set_palette_to_rgb(pngobj);

        if(png_get_valid( pngobj, info, PNG_INFO_tRNS ) ) {
            hasAlpha = true;
        }
    }

    unsigned char **lines =
        (unsigned char **)malloc(sizeof(unsigned char *) * height);

    if( hasAlpha ) {
        // to 32bit image
        image.init( width, height, Image::TYPE_32BIT );
    } else {
        // to 24bit image
        image.init( width, height, Image::TYPE_24BIT );
    }

    for( int i=0; i<height; i++ ) {
        // Load upside down
        lines[i] = (unsigned char*)image.getLineBuffer(height-1-i);
    }

    png_read_image( pngobj, lines );
    png_read_end( pngobj, NULL );

    free( lines );

    png_destroy_read_struct( &pngobj, &info, NULL );
    
    return true;
}
