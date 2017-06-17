#include <math.h>
#include "TrackBall.h"

#define TRANS_RATE        (0.15f)
#define ZOOM_RATE         (0.04f)

static void setQuatFromAxisAngle(Quat4f& q,
								 const Vector4f& axis,
								 float angle) {
	const float halfAngle = angle * 0.5f;

	float sin = sinf(halfAngle);
	float cos = cosf(halfAngle);

	q.set(sin * axis.x, sin * axis.y, sin * axis.z, cos);

	if( q.lengthSquared() == 0.0f) {
		q.set(0.0f, 0.0f, 0.0f, 1.0f);
	}
}

/**
 * TrackBall():
 *
 * pitchAngle = 俯角(+だと上を向く方向)
 */
TrackBall::TrackBall(float eyex_, float eyey_, float eyez_, 
					 float pitchAngle) {
	eyex = eyex_;
	eyey = eyey_;
	eyez = eyez_;

	lastX = lastY = 0;
	trans.setIdentity();

	const Vector4f axisX(1.0f, 0.0f, 0.0f, 0.0f);

	setQuatFromAxisAngle(q, axisX, pitchAngle);
	width = height = 0;
}

/**
 * resize():
 */
void TrackBall::resize(int width_, int height_) {
	width = width_;
	height = height_;
}

/**
 * getMat():
 */
void TrackBall::getMat(Matrix4f& mat) const {
	Matrix4f eyeTrans;
	eyeTrans.setIdentity();
	eyeTrans.setColumn(3, Vector4f(eyex, eyey, eyez, 1.0f));

	mat.set(q);
	mat *= trans;
	mat *= eyeTrans;
}

/**
 * startRotation():
 */
void TrackBall::startRotation(int x, int y) {
	// ドラッグの最初に球面上の点lastposを出しておく
	sphericalMap(x, y, lastSpherialPos);
}

/**
 * startZoom():
 */
void TrackBall::startZoom(int x, int y) {
	lastX = x;
	lastY = y;
}

/**
 * startTrans():
 */
void TrackBall::startTrans(int x, int y) {
	lastX = x;
	lastY = y;
}

/**
 * dragRotation():
 */
void TrackBall::dragRotation(int x, int y) {
	// x, yを球面上の点にマッピングし、
	// 前回のマッピング点との距離および外積から
	// axisを出す.
	// axis[]軸にangle度回転したものと、qを
	// かけて新しいqにする.
	Vector4f axis;
	float angle;
	calcAxisAngle(x, y, axis, angle);

	axis *= -3.0f;

	Quat4f dq;
	setQuatFromAxisAngle(dq, axis, angle);
	q *= dq;
	q.normalize();
}

/**
 * dragZoom():
 */
void TrackBall::dragZoom(int x, int y) {
	// translationマトリクスを更新する.
	float len = ZOOM_RATE * (float)(y - lastY);

	float transZ = trans.getElement(2, 3);
	transZ -= len;
	trans.setElement(2, 3, transZ);

	lastX = x;
	lastY = y;
}

/**
 * dragTrans():
 */
void TrackBall::dragTrans(int x, int y) {
	float dx = TRANS_RATE * float(x - lastX);
	float dy = TRANS_RATE * float(y - lastY);

	float tx = trans.getElement(0, 3);
	float ty = trans.getElement(1, 3);

	tx -= dx;
	ty -= dy;

	trans.setElement(0, 3, tx);
	trans.setElement(1, 3, ty);
	
	lastX = x;
	lastY = y;
}

/**
 * calcAxisAngle():
 *
 * x, yをsphericalMapにて球上の点(curpos)に変換し、
 * lastposとの距離によりangleを、
 * lastposとの外積によりaxisを求める.
 */
void TrackBall::calcAxisAngle(int x, int y, 
							  Vector4f& axis,
							  float& angle) {

	Vector4f curSphericalPos;
	Vector4f dv;

	sphericalMap(x, y, curSphericalPos);
	dv.sub(curSphericalPos, lastSpherialPos);
	
	angle = PI * 0.5f * dv.length();
	axis.cross(lastSpherialPos, curSphericalPos);

	lastSpherialPos = curSphericalPos;
}

/**
 * sphericalMap():
 * 
 * [x, yを半径1.0の球上の3次元点に対応させる]
 *
 *      -1.0x        +1.0x
 *        +-----+-----+   1.0y
 *        +     +     +
 *        +-----+-----+
 *        +     +     +
 *        +-----+-----+  -1.0y
 *
 *    手前+z方向で、手前に半球面があるイメージ
 */
void TrackBall::sphericalMap(int x, int y, Vector4f& v) {
	// windowsの上端がy=0で来る.
	float d;
	v.x =  (2.0f * x - width)  / width;
	v.y = -(2.0f * y - height) / height;
	d = sqrtf(v.x*v.x + v.y*v.y);
	if(d >= 1.0f) {
		d = 1.0f;
	}
	v.z = cosf(PI * 0.5f * d);
	v.w = 0.0f;

	v.normalize();
}
