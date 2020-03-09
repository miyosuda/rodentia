// -*- C++ -*-
// ----------------------------------------------
// This LiSPSM code is based on this web article.
// http://asura.iaigiri.com/OpenGL/gl59.html
// ----------------------------------------------

#ifndef LSPSM_HEADER
#define LSPSM_HEADER

#include "Matrix4f.h"
#include "Vector3f.h"

class BoundingBox;

class VolumePoints {
private:
    static const int POINT_SIZE = 8;
    Vector3f points[POINT_SIZE];

public:
    VolumePoints();
    void init();
    void transform(const Matrix4f& matrix);
    void computeBoundingBox(BoundingBox& boundingBox) const;
    VolumePoints& operator=(const VolumePoints &value);
    void debugDump() const;
};

class LSPSM {
private:
    // Camera position
    Vector3f eyePosition;
    // Camera eye direction
    Vector3f viewDir;
    // Light direction
    Vector3f lightDir;

    // Camera view matrix
    Matrix4f eyeViewMat;
    // Camera projection matrix
    Matrix4f eyeProjMat;
    // Light view marix
    Matrix4f lightViewMat;
    // Light projection matrix
    Matrix4f lightProjMat;
    // Light view proection matrix (=result)
    Matrix4f lightViewProjMat;

    // distance to near clip plane
    float nearClip;

    void computeUpVector(const Vector3f& viewDir,
                         const Vector3f& lightDir,
                         Vector3f& up);
    void computeMatrix_USM(VolumePoints& points);
    void computeMatrix_LSPSM(float angle, VolumePoints& points);
    void computeLightVolumePoints(const Matrix4f& viewProj, VolumePoints& points);
    void computeViewFrustum(const Matrix4f& viewProj, VolumePoints& points);
    void getUnitCubeClipMatrix(const BoundingBox& boundingBox, Matrix4f& mat) const;
    void getPerspective(const float nearDist,
                        const float farDist,
                        Matrix4f& mat) const;

public:
    LSPSM();
    
    void updateShadowMatrix();

    void setEyePos(const Vector3f& value) {
        eyePosition.set(value);
    }

    void setViewDir(const Vector3f& value) {
        viewDir.set(value);
        viewDir.normalize();
    }

    void setLightDir(const Vector3f& value) {
        lightDir.set(value);
        lightDir.normalize();
    }

    void setEyeView(const Matrix4f& value) {
        eyeViewMat.set(value);
    }

    void setEyeProjection(const Matrix4f& value) {
        eyeProjMat.set(value);
    }

    void setNearClip(float value) {
        nearClip = value;
    }

    const Matrix4f& getLightView() {
        return lightViewMat;
    }

    const Matrix4f& getLightProjection() {
        return lightProjMat;
    }

    const Matrix4f& getLightViewProjection() {
        return lightViewProjMat;
    }
};

#endif // __LSPSM_H__
