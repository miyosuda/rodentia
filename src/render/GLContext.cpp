#include "GLContext.h"
#include <stdio.h>

#if defined(__APPLE__)

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
bool GLContext::init(int /*width*/, int /*height*/) {
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
        printf("Failed: CGLChoosePixelFormat: error code=%d\n", errorCode);
        return false;
    }
    
    errorCode = CGLCreateContext(pixelFormatObj, NULL, &context);
    if( errorCode != 0 ) {
        printf("Failed: CGLCreateContext: error code=%d\n", errorCode);
        CGLDestroyPixelFormat(pixelFormatObj);
        return false;
    }
    
    CGLDestroyPixelFormat(pixelFormatObj);
    
    errorCode = CGLSetCurrentContext(context);
    if( errorCode != 0 ) {
        printf("Failed: CGLSetCurrentContext: error code=%d\n", errorCode);
        return false;
    }

    contextInitialized = true;

    // Load glad
    bool ret = gladLoadGL();
    if( !ret ) {
        printf("Failed to init glad.\n");
        return false;
    }

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

#else // defined(__APPLE__)

/**
 * <!--  GLContext():  -->
 */
GLContext::GLContext()
    :
    display(NULL),
    context(0),
    pbuffer(0),
    contextInitialized(false) {
}

typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);

/**
 * <!--  init():  -->
 */
bool GLContext::init(int width, int height) {
    glXCreateContextAttribsARBProc glXCreateContextAttribsARB =
        (glXCreateContextAttribsARBProc)glXGetProcAddressARB((const GLubyte*)"glXCreateContextAttribsARB");

    const char *displayName = NULL;
    display = XOpenDisplay(displayName);

    static int visualAttribs[] = { None };
    int numberOfFramebufferConfigurations = 0;
    GLXFBConfig* fbConfigs = glXChooseFBConfig(display,
                                               DefaultScreen(display),
                                               visualAttribs,
                                               &numberOfFramebufferConfigurations);

    int context_attribs[] = {
        GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
        GLX_CONTEXT_MINOR_VERSION_ARB, 2,
        GLX_CONTEXT_FLAGS_ARB, GLX_CONTEXT_DEBUG_BIT_ARB,
        GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
        None
    };

    context = glXCreateContextAttribsARB(display,
                                         fbConfigs[0],
                                         0,
                                         True,
                                         context_attribs);

    int pbufferAttribs[] = {
        GLX_PBUFFER_WIDTH, width,
        GLX_PBUFFER_HEIGHT, height,
        None
    };
    pbuffer = glXCreatePbuffer(display, fbConfigs[0], pbufferAttribs);
 
    XFree(fbConfigs);
    XSync(display, False);
 
    if( !glXMakeContextCurrent( display, pbuffer, pbuffer, context) ) {
        printf("Failed to GLX context make current");
        return false;
    }
    
    contextInitialized = true;

    // Load glad
    bool ret = gladLoadGL();
    if( !ret ) {
        printf("Failed to init glad.\n");
        return false;
    }
    
    return true;
}
    
/**
 * <!--  release():  -->
 */
void GLContext::release() {
    if( contextInitialized ) {
        glXMakeContextCurrent(display, 0, 0, 0);
        // TODO: glxDestroyPbuffer function not found?
        //glxDestroyPbuffer(display, pbuffer);
        glXDestroyContext(display, context);
        XCloseDisplay(display);

        display = NULL;
        context = 0;        
        pbuffer = 0;
    }
}

#endif // defined(__APPLE__)
