#include "CameraView.h"

/**
 * <!--  CameraView():  -->
 */
CameraView::CameraView() {
}

/**
 * <!--  init():  -->
 */
bool CameraView::init(int width, int height, const Vector3f& bgColor,
                      float nearClip, float farClip, float focalLength,
                      int shadowBufferWidth) {
    // Setup caemra
    float ratio = width / (float) height;

    const bool flipping = true;
    camera.initPerspective(nearClip, farClip, focalLength, ratio, flipping);

    // Setup render target
    bool ret = renderTarget.init(width, height, bgColor, shadowBufferWidth);
    return ret;
}

/**
 * <!--  release():  -->
 */
void CameraView::release() {
    renderTarget.release();
}

/**
 * <!--  prepareShadowDepthRendering():  -->
 */
void CameraView::prepareShadowDepthRendering() {
    renderTarget.prepareShadowDepthRendering();
}

/**
 * <!--  prepareRendering():  -->
 */
void CameraView::prepareRendering() {
    renderTarget.prepareRendering();
}

/**
 * <!--  finishRendering():  -->
 */
void CameraView::finishRendering() {
    renderTarget.finishRendering();
}
