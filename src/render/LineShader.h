// -*- C++ -*-
#ifndef LINESHADER_HEADER
#define LINESHADER_HEADER

#include "Shader.h"

class LineShader : public Shader {
private:
	int vertexHandle;
	int mvpMatrixHandle;
	int lineColorHandle;
	
public:
	virtual bool init() override;
	virtual void setMatrix(const Matrix4f& modelViewMat,
						   const Matrix4f& modelViewProjectionMat)
		const override;
	virtual void beginRender(const float* vertices) const override;
	virtual void render(const short* indices, int indicesSize) const override;
	virtual void endRender() const override;
	virtual void setColor(const Vector4f& color) const override;
};

#endif
