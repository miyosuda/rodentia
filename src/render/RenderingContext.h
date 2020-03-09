// -*- C++ -*-
#ifndef RENDERINGCONTEXT_HEADER
#define RENDERINGCONTEXT_HEADER

#include "Matrix4f.h"
#include "Camera.h"
#include "Vector3f.h"

#include "LSPSM.h"

class BoundingBox;


class RenderingContext {
public: 
    enum Path {
        SHADOW, // rendering shadow depth (1st path)
        NORMAL, // noral rendering (2nd path)
    };
    
private:
    Path path;

    Matrix4f cameraInvMat;
    Matrix4f cameraProjectionMat;
    
    // Directional light direction.
    Vector3f lightDir;
    // Light color
    Vector4f lightColor;
    // Ambient color
    Vector4f ambientColor;
    // Shadow color rate
    float shadowColorRate;
    
    // Model matrix for current drawing object.
    Matrix4f modelMat;
    // Cached model view matrix
    Matrix4f modelViewMat;
    // Cached model view projection matrix
    Matrix4f modelViewProjectionMat;

    // Matrix to convert viewport coord(-1~1) to textue coord. (0~1)
    const Matrix4f depthBiasMat;
    // Model view projection matrix for depth from light view.
    Matrix4f depthModelViewProjectionMat;
    // Matrix of (depthBiasMat * depthModelViewProjectionMat)
    Matrix4f depthBiasModelViewProjectionMat;

    LSPSM lspsm;
    void updateLSPSM();

public:
    RenderingContext();
    void setCamera(const Matrix4f& cameraMat,
                   const Matrix4f& cameraInvMat_,
                   const Matrix4f& cameraProjectionMat_);
    void setModelMat(Matrix4f modelMat_);
    void setLight(const Vector3f& lightDir_,
                  const Vector3f& lightColor_,
                  const Vector3f& ambientColor_,
                  float shadowColorRate_);

    void setPath(Path path_);
    bool isRenderingShadow() const { return path == SHADOW; }
    void setBoundingBoxForShadow(const BoundingBox& boundingBox);

    const Vector3f& getLightDir() const {
        return lightDir;
    }
    const Vector4f& getLightColor() const {
        return lightColor;
    }
    const Vector4f& getAmbientColor() const {
        return ambientColor;
    }
    float getShadowColorRate() const {
        return shadowColorRate;
    }
    const Matrix4f& getModelMat() const {
        return modelMat;
    }
    const Matrix4f& getModelViewMat() const {
        return modelViewMat;
    }
    const Matrix4f& getModelViewProjectionMat() const {
        return modelViewProjectionMat;
    }
    const Matrix4f& getDepthModelViewProjectionMat() const {
        return depthModelViewProjectionMat;
    }
    const Matrix4f& getDepthBiasModelViewProjectionMat() const {
        return depthBiasModelViewProjectionMat;
    }
};

#endif
