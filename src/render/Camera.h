// -*- C++ -*-
#ifndef CAMERA_HEADER
#define CAMERA_HEADER

#include "Matrix4f.h"

class Vector3f;


class Camera {
private:
    Matrix4f mat;
    Matrix4f invMat;
    Matrix4f projectionMat;
    
    float znear; // distance to znear clip plane
    float nearWidth; // znear clip place width
    
public:
    Camera();
    void initPerspective(float znear_, float zfar, float focalLength, float ratio,
                         bool flipping);
    //void initOrtho(float znear_, float zfar, float width, float height);
    void initOrtho(float znear_, float zfar,
                   float left, float right, float bottom, float top);

    void setMat(const Matrix4f& mat_);
    const Matrix4f& getMat() const {
        return mat;
    }
    const Matrix4f& getInvMat() const {
        return invMat;
    }
    const Matrix4f& getProjectionMat() const {
        return projectionMat;
    }
    
    void lookAt( const Vector3f& fromPos,
                 const Vector3f& toPos,
                 const Vector3f& up );
};


#endif
