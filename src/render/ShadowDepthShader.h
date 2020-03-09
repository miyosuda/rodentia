// -*- C++ -*-
#ifndef SHADOWDEPTHSHADER_HEADER
#define SHADOWDEPTHSHADER_HEADER

#include "Shader.h"


class ShadowDepthShader : public Shader {
private:
    int mvpMatrixHandle;
    
public:
    virtual bool init() override;
    virtual void prepare(const RenderingContext& context) const override;
    virtual void setup(const RenderingContext& context) const override;
};

#endif
