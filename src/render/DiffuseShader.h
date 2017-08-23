// -*- C++ -*-
#ifndef DIFFUSESHADER_HEADER
#define DIFFUSESHADER_HEADER

#include "Shader.h"

class Matrix4f;
class Vector3f;

class DiffuseShader : public Shader {
private:
	int vertexHandle;
	int normalHandle;
	int textureCoordHandle;
	int mvpMatrixHandle;
	int normalMatrixHandle;
	int directionalLightDir0Handle;
	
public:
	virtual bool init() override;
	virtual void setMatrix(const Matrix4f& modelMat,
						   const Matrix4f& modelViewMat,
						   const Matrix4f& modelViewProjectionMat)
		const override;
	virtual void beginRender(const float* vertices) const override;
	virtual void render(const unsigned short* indices, int indicesSize) const override;
	virtual void endRender() const override;
	virtual void setDirectionalLight(const Vector3f& lightDir) const override;
};

#endif
