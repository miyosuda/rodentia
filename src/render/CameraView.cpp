#include "CameraView.h"

/**
 * <!--  CameraView():  -->
 */
CameraView::CameraView() {
}

/**
 * <!--  init():  -->
 */
bool CameraView::init(int width, int height, const Vector3f& bgColor_) {
    // Setup caemra
    float ratio = width / (float) height;

    const float nearClip = 0.05f;
    const float farClip = 80.0f;
    const float focalLength = 50.0f;
    const bool flipping = true;
    camera.initPerspective(nearClip, farClip, focalLength, ratio, flipping);

    // Setup render target
    bool ret = renderTarget.init(width, height, bgColor_);
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
