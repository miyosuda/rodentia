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
	int invLightDirHandle;
	
public:
	virtual bool init() override;
	virtual void prepare(const RenderingContext& context) const override;
	virtual void setup(const RenderingContext& context) const override;
	virtual void beginRender(const float* vertices) const override;
	virtual void render(const unsigned short* indices, int indicesSize) const override;
	virtual void endRender() const override;
};

#endif
