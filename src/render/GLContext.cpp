#include "GLContext.h"
#include <stdio.h>

/**
 * <!--  GLContext():  -->
 */
GLContext::GLContext()
	:
	contextInitialized(false) {
}

/**
 * <!--  init():  -->
 */
bool GLContext::init() {
	CGLPixelFormatAttribute attributes[4] = {
		kCGLPFAAccelerated,
		kCGLPFAOpenGLProfile,		
		(CGLPixelFormatAttribute) kCGLOGLPVersion_3_2_Core,
		(CGLPixelFormatAttribute) 0
	};

	CGLError errorCode;
	CGLPixelFormatObj pixelFormatObj;
	GLint numPixelFormats;
	
	errorCode = CGLChoosePixelFormat(attributes, &pixelFormatObj, &numPixelFormats);
	if( errorCode != 0 ) {
		printf("Failed: CGLChoosePixelFormat\n");
		return false;
	}
	
	errorCode = CGLCreateContext(pixelFormatObj, NULL, &context);
	if( errorCode != 0 ) {
		printf("Failed: CGLCreateContext\n");
		CGLDestroyPixelFormat(pixelFormatObj);
		return false;
	}
	
	CGLDestroyPixelFormat(pixelFormatObj);
	
	errorCode = CGLSetCurrentContext(context);
	if( errorCode != 0 ) {
		printf("Failed: CGLSetCurrentContext\n");
		return false;
	}

	contextInitialized = true;

	return true;
}
	
/**
 * <!--  release():  -->
 */
void GLContext::release() {
	if( contextInitialized ) {
		CGLSetCurrentContext(NULL);
		CGLDestroyContext(context);
	}
}
