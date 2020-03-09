#include "Camera.h"
#include "Vector3f.h"

/**
 * <!--  setProjectionMatrix():  -->
 */
static void setProjectionMatrix(Matrix4f& m,
                                float w, float h,
                                float near, float far,
                                bool flipping) {
    float a = 2.0f * near / w;
    float b = 2.0f * near / h;

    if( flipping ) {
        b = -b;
    }
    
    float c = - (far + near) / (far - near);
    float d = -2.0f * far * near / (far - near);
    
    m.m00 = a; m.m01 = 0; m.m02 = 0;  m.m03 = 0;
    m.m10 = 0; m.m11 = b; m.m12 = 0;  m.m13 = 0;
    m.m20 = 0; m.m21 = 0; m.m22 = c;  m.m23 = d;
    m.m30 = 0; m.m31 = 0; m.m32 = -1; m.m33 = 0;
}

/**
 * <!--  Camera():  -->
 */
Camera::Camera() {
    Matrix4f m;
    m.setIdentity();
    setMat(m);
}

/**
 * @param focalLength_ focul length based on 35.0mm film
 * @param ratio        w/h (when portrait ratio is bigger than 1.0)
 * @param flipping     if true render upside down
 */
void Camera::initPerspective(float znear_, float zfar_, float focalLength, float ratio,
                             bool flipping) {
    znear = znear_;
    float h = znear * 35.0f / focalLength;
    float w = h * ratio;
    setProjectionMatrix(projectionMat, w, h, znear, zfar_, flipping);

    nearWidth = w;
}

/**
 * <!--  initOrtho():  -->
 */
void Camera::initOrtho(float znear_, float zfar,
                       float left, float right, float bottom, float top) {
    znear = znear_;
    nearWidth = right - left;

    float sx =  2.0f / (right - left);
    float sy =  2.0f / (top - bottom);
    float sz = -2.0f / (zfar - znear);
    float tx = -(right+left) / (right-left);
    float ty = -(top+bottom) / (top-bottom);
    float tz = -(zfar+znear) / (zfar-znear);

    Matrix4f& m = projectionMat;
    m.m00 = sx;    m.m01 = 0.0f;  m.m02 = 0.0f;  m.m03 = tx;
    m.m10 = 0.0f;  m.m11 = sy;    m.m12 = 0.0f;  m.m13 = ty;
    m.m20 = 0.0f;  m.m21 = 0.0f;  m.m22 = sz;    m.m23 = tz;
    m.m30 = 0.0f;  m.m31 = 0.0f;  m.m32 = 0.0f;  m.m33 = 1.0f;
}

/**
 * <!--  setMat():  -->
 */
void Camera::setMat(const Matrix4f& mat_) {
    mat.set(mat_);
    invMat.invertRT(mat);
}

/**
 * <!--  lookAt():  -->
 */
void Camera::lookAt( const Vector3f& fromPos,
                     const Vector3f& toPos,
                     const Vector3f& up ) {
    Vector3f forward;
    forward.sub(toPos, fromPos);
    forward.normalize();

    Vector3f side;
    side.cross(forward, up);

    if( side.lengthSquared() < 0.00001f ) {
        side.set(1.0f, 0.0f, 0.0f);
    } else {
        side.normalize();
    }

    Vector3f newUp;
    newUp.cross(side, forward);

    mat.setZero();
    
    mat.m00 = side.x;
    mat.m10 = side.y;
    mat.m20 = side.z;

    mat.m01 = newUp.x;
    mat.m11 = newUp.y;
    mat.m21 = newUp.z;

    mat.m02 = -forward.x;
    mat.m12 = -forward.y;
    mat.m22 = -forward.z;

    mat.m03 = fromPos.x;
    mat.m13 = fromPos.y;
    mat.m23 = fromPos.z;
    
    mat.m33 = 1.0f;
    
    invMat.invertRT(mat);
}
