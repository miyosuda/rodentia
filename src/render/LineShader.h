// -*- C++ -*-
#ifndef LINESHADER_HEADER
#define LINESHADER_HEADER

#include "Shader.h"

class LineShader : public Shader {
private:
    int mvpMatrixHandle;
    //int lineColorHandle;
    
public:
    virtual bool init() override;
    virtual void setup(const RenderingContext& context) const override;
    //virtual void setColor(const Vector4f& color) const override;
};

#endif
