#include "Renderer.h"

/**
 * <!--  Renderer():  -->
 */
Renderer::Renderer() {
}

/**
 * <!--  init():  -->
 */
bool Renderer::init(int width, int height, const Vector3f& bgColor_) {
    bool ret;
    ret = context.init(width, height);
    
    if( !ret ) {
        printf("Failed to initialize GL context\n");
        return false;
    }

    ret = renderTarget.init(width, height, bgColor_);
    return ret;
}

/**
 * <!--  release():  -->
 */
void Renderer::release() {
    renderTarget.release();
    context.release();
}

/**
 * <!--  prepareShadowDepthRendering():  -->
 */
void Renderer::prepareShadowDepthRendering() {
    renderTarget.prepareShadowDepthRendering();
}

/**
 * <!--  prepareRendering():  -->
 */
void Renderer::prepareRendering() {
    renderTarget.prepareRendering();
}

/**
 * <!--  finishRendering():  -->
 */
void Renderer::finishRendering() {
    renderTarget.finishRendering();
}
