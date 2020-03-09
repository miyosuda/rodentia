#include "RenderTarget.h"

#include <stdlib.h>
#include <stdio.h>

#include "glinc.h"

/**
 * <!--  calcDepthFrameBufferSize():  -->
 *
 * Calculate size of depth buffer for shadow mapping based on view width and height.
 */
int RenderTarget::calcDepthFrameBufferSize(int width, int height) {
    int minSize = width;
    if( height < minSize ) {
        minSize = height;
    }

    int result = 64;
    while(true) {
        if( result >= minSize ) {
            break;
        }
        result *= 2;
    }
    if( result > 1024 ) {
        // Max shadow map depth buffer size is 1024x1024 
        result = 1024;
    }

    return result;
}


/**
 * <!--  init():  -->
 */
bool RenderTarget::init(int width, int height, const Vector3f& bgColor_,
                        int shadowBufferWidth) {
    bgColor.set(bgColor_);
    
    frameBufferWidth = width;
    frameBufferHeight = height;

    // Setup rendering frame buffer
    bool ret = frameBuffer.init(frameBufferWidth, frameBufferHeight);
    
    if( !ret ) {
        printf("Failed to init offscreen frame buffer.\n");
        return false;
    }

    // Setup shadow depth frame buffer
    int depthFrameBufferSize;
    if( shadowBufferWidth <= 0 ) {
        // Calculate depth frame buffer size automatically.
        depthFrameBufferSize = calcDepthFrameBufferSize(width, height);
    } else {
        depthFrameBufferSize = shadowBufferWidth;
    }
    
    ret = depthFrameBuffer.init(depthFrameBufferSize, depthFrameBufferSize);
    if( !ret ) {
        printf("Failed to init shadow depth buffer.\n");
        return false;
    }
    
    buffer = calloc(4, getFrameBufferSize()/4);
    
    return true;
}

/**
 * <!--  release():  -->
 */
void RenderTarget::release() {
    if( buffer != nullptr ) {
        free(buffer);
        buffer = nullptr;
    }
}

/**
 * <!--  prepareShadowDepthRendering():  -->
 */
void RenderTarget::prepareShadowDepthRendering() {
    depthFrameBuffer.use();
    
    depthFrameBuffer.setViewport();

    glFrontFace(GL_CCW); // Default
    glEnable(GL_CULL_FACE); 
    glCullFace(GL_FRONT); // Cull front face
    
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    
    glClear(GL_DEPTH_BUFFER_BIT);
}

/**
 * <!--  prepareRendering():  -->
 */
void RenderTarget::prepareRendering() {
    frameBuffer.use();
    
    frameBuffer.setViewport();

    glFrontFace(GL_CW); // Flipped because camera is inverted as upside down
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK); // Cull back face
    
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    
    glClearColor(bgColor.x, bgColor.y, bgColor.z, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Set depth frame buffer for texture slot 1
    glEnable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE1);
    depthFrameBuffer.bind();
}

/**
 * <!--  finishRendering():  -->
 */
void RenderTarget::finishRendering() {
    glReadPixels(0, 0, frameBufferWidth, frameBufferHeight, GL_RGB, GL_UNSIGNED_BYTE, buffer);

    // TODO: WORKAROUND
    // Resetting gl error here
    /*int error = */glGetError();
}
