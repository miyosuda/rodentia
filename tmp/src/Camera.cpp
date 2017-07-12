#include "Camera.h"

/**
 * <!--  setProjectionMatrix():  -->
 */
static void setProjectionMatrix(Matrix4f& m,
								float w, float h,
								float near, float far) {
	float a = 2.0f * near / w;
	float b = 2.0f * near / h;
	float c = - (far + near) / (far - near);
	float d = -2.0f * far * near / (far - near);
	
	m.m00 = a; m.m01 = 0; m.m02 = 0;  m.m03 = 0;
	m.m10 = 0; m.m11 = b; m.m12 = 0;  m.m13 = 0;
	m.m20 = 0; m.m21 = 0; m.m22 = c;  m.m23 = d;
	m.m30 = 0; m.m31 = 0; m.m32 = -1; m.m33 = 1;
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
 * @param ratio  w/h (when portrait ratio is bigger than 1.0)
 */
void Camera::init(float znear_, float zfar_, float focalLength, float ratio) {
	znear = znear_;
	float h = znear * 35.0f / focalLength;
	float w = h * ratio;
	setProjectionMatrix(projectionMat, w, h, znear, zfar_);

	nearWidth = w;
}

/**
 * <!--  setMat():  -->
 */
void Camera::setMat(const Matrix4f& mat_) {
	mat.set(mat_);
	invMat.invertRT(mat);
}
