// -*- C++ -*-
#ifndef SHADOWDIFFUSESHADER_HEADER
#define SHADOWDIFFUSESHADER_HEADER

#include "Shader.h"

class ShadowDiffuseShader : public Shader {
private:
	int mvpMatrixHandle;
	int depthBiasMvpMatrixHandle;
	int normalMatrixHandle;
	int invLightDirHandle;
	int textureHandle;
	int shadowMapHandle;
	
public:
	virtual bool init() override;
	virtual void prepare(const RenderingContext& context) const override;
	virtual void setup(const RenderingContext& context) const override;
};

#endif
