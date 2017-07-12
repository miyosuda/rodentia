#include "Camera.h"

//-----------------------------------
//         [RenderManager]
//-----------------------------------

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

Camera::Camera() {
	head = 0.0f;
	pitch = 0.0f;
	setPos(0.0f, 0.0f, 0.0f);
	matdirty = true;
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
	
const Vector4f& Camera::getPos() const {
	return pos;
}
	
void Camera::setPos(float x, float y, float z) {
	pos.set(x,y,z, 1.0f);
	matdirty = true;
}
	
void Camera::setPos(Vector4f pos_) {
	pos.set(pos_);
	matdirty = true;
}
	
void Camera::update() {
	if( matdirty ) {
		mat.setIdentity();
		mat.setRotationY(head);
		Matrix4f mat0;
		mat0.setRotationX(pitch);
		mat *= mat0;
		mat.setColumn(3, Vector4f(pos.x, pos.y, pos.z, 1.0f));
		
		invMat.invertRT(mat);
		matdirty = false;
	}
}

void Camera::setHead(float head_) {
	head = head_;
	matdirty = true;
}
	
void Camera::setPitch(float pitch_) {
	pitch = pitch_;
	matdirty = true;
}

void Camera::addHead(float dh) {
	head += dh;
	matdirty = true;
}

void Camera::addPitch(float dp) {
	pitch += dp;
	matdirty = true;
}
