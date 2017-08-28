// -*- C++ -*-
#ifndef RENDERINGCONTEXT_HEADER
#define RENDERINGCONTEXT_HEADER

#include "Matrix4f.h"
#include "Camera.h"
#include "Vector3f.h"


class RenderingContext {
private:
	Camera camera;

	Vector3f lightDir;
	Matrix4f modelMat;               // model matrix for current drawing object.
	Matrix4f modelViewMat;           // Cached model view matrix
	Matrix4f modelViewProjectionMat; // Cached model view projection matrix

public:
	RenderingContext();
	
	void initCamera(float ratio, bool flipping);
	void setModelMat(Matrix4f modelMat_);
	void setCameraMat(const Matrix4f& mat);
	void setLightDir(const Vector3f& lightDir_); 

	//const Camera& getCamera() const { return camera; }

	const Vector3f& getLightDir() const {
		return lightDir;
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
};

#endif
