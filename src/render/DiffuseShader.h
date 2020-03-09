// -*- C++ -*-
#ifndef DIFFUSESHADER_HEADER
#define DIFFUSESHADER_HEADER

#include "Shader.h"

class DiffuseShader : public Shader {
private:
    int mvpMatrixHandle;
    int normalMatrixHandle;
    int invLightDirHandle;
    int lightColorHandle;
    int ambientColorHandle;
    
public:
    virtual bool init() override;
    virtual void prepare(const RenderingContext& context) const override;
    virtual void setup(const RenderingContext& context) const override;
};

#endif
