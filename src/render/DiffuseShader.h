// -*- C++ -*-
#ifndef DIFFUSESHADER_HEADER
#define DIFFUSESHADER_HEADER

#include "Shader.h"

class Matrix4f;
class Vector3f;

class DiffuseShader : public Shader {
private:
	int mvpMatrixHandle;
	int normalMatrixHandle;
	int invLightDirHandle;
	
public:
	virtual bool init() override;
	virtual void prepare(const RenderingContext& context) const override;
	virtual void setup(const RenderingContext& context) const override;
};

#endif
