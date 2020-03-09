// -*- C++ -*-
#ifndef CAMERAVIEW_HEADER
#define CAMERAVIEW_HEADER

#include "Matrix4f.h"
#include "Vector3f.h"
#include "Camera.h"
#include "RenderTarget.h"


class CameraView {
private:
    Camera camera;
    RenderTarget renderTarget;

public:
    CameraView();
    bool init(int width, int height, const Vector3f& bgColor,
              float nearClip, float farClip, float focalLength,
              int shadowBufferWidth);
    void release();

    void setCameraMat(const Matrix4f& mat) {
        camera.setMat(mat);
    }

    const Matrix4f& getCameraMat() const {
        return camera.getMat();
    }
    
    const Matrix4f& getCameraInvMat() const {
        return camera.getInvMat();
    }

    const Matrix4f& getCameraProjectionMat() const {
        return camera.getProjectionMat();
    }

    int getFrameBufferWidth()  const { return renderTarget.getFrameBufferWidth();  }
    int getFrameBufferHeight() const { return renderTarget.getFrameBufferHeight(); }
    
    const void* getBuffer() const {
        return renderTarget.getBuffer();
    }
    
    int getFrameBufferSize() const {
        return renderTarget.getFrameBufferSize();
    }
    
    void prepareShadowDepthRendering();
    void prepareRendering();
    void finishRendering();
};

#endif
