// -*- C++ -*-
#ifndef CAMERA_HEADER
#define CAMERA_HEADER

#include "Matrix4f.h"

class Camera {
private:
	Matrix4f mat;
	Matrix4f invMat;
	Matrix4f projectionMat;
	
	float znear; // distance to znear clip plane
	float nearWidth; // znear clip place width
	
public:
	Camera();
	void init(float znear_, float zfar_, float focalLength, float ratio, bool flipping);

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
};


#endif
