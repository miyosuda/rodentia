// -*- C++ -*-
#ifndef DIFFUSESHADER_HEADER
#define DIFFUSESHADER_HEADER

#include "Shader.h"

class Matrix4f;

class DiffuseShader : public Shader {
private:
	int vertexHandle;
	int normalHandle;
	int textureCoordHandle;
	int mvpMatrixHandle;
	int normalMatrixHandle;
	
public:
	virtual bool init() override;
	virtual void setMatrix(const Matrix4f& modelViewMat,
						   const Matrix4f& modelViewProjectionMat) override;
	virtual void beginRender(const float* vertices) override;
	virtual void render(const short* indices, int indicesSize) override;
	virtual void endRender() override;
};

#endif
