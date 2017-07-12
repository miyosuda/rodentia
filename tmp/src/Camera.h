// -*- C++ -*-
#ifndef CAMERA_HEADER
#define CAMERA_HEADER

#include "Matrix4f.h"
#include "Vector4f.h"

class Camera {
private:
	Matrix4f mat;
	Matrix4f invMat;
	Matrix4f projectionMat;
	
	Vector4f pos;
	float head;
	float pitch;
	bool matdirty;

	float znear; // distance to znear clip plane
	float nearWidth; // znear clip place width
	
public:
	Camera();
	void init(float znear_, float zfar_, float focalLength, float ratio);
	const Vector4f& getPos() const;
	void setPos(float x, float y, float z);
	void setPos(Vector4f pos_);
	void update();
	void setHead(float head_);
	void setPitch(float pitch_);
	void addHead(float dh);
	void addPitch(float dp);
	float getHead() const { return head; }
	float getPitch() const { return pitch; }

	const Matrix4f& getMat() const {
		return mat;
	}
	const Matrix4f& getInvMat() const {
		return invMat;
	}
	const Matrix4f& getProjectionMat() const {
		return projectionMat;
	}

};


#endif
