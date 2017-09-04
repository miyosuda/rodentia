// -*- C++ -*-
#ifndef RENDERINGCONTEXT_HEADER
#define RENDERINGCONTEXT_HEADER

#include "Matrix4f.h"
#include "Camera.h"
#include "Vector3f.h"


class RenderingContext {
public:	
	enum Path {
		SHADOW, // rendering shadow depth (1st path)
		NORMAL, // noral rendering (2nd path)
	};
	
private:
	Path path;
	Camera camera;
	Camera lightCamera;

	// Directional light direction.
	Vector3f lightDir;
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

public:
	RenderingContext();
	
	void initCamera(float ratio, bool flipping=true);
	void setModelMat(Matrix4f modelMat_);
	void setCameraMat(const Matrix4f& mat);
	void setLightDir(const Vector3f& lightDir_);

	void setPath(Path path_);
	bool isRenderingShadow() const { return path == SHADOW; }

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
	const Matrix4f& getDepthModelViewProjectionMat() const {
		return depthModelViewProjectionMat;
	}
	const Matrix4f& getDepthBiasModelViewProjectionMat() const {
		return depthBiasModelViewProjectionMat;
	}
};

#endif
