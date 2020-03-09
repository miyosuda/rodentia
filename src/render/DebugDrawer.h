// -*- C++ -*-
#ifndef DEBUGDRAWER_HEADER
#define DEBUGDRAWER_HEADER

#include "btBulletDynamicsCommon.h"

#include "BufferObjects.h"

class Shader;
class Matrix4f;
class RenderingContext;


class DebugDrawer: public btIDebugDraw {
private:
    int debugMode;
    Shader* lineShader;

    VertexArray vertexArray;
    VertexBuffer vertexBuffer;
    IndexBuffer indexBuffer;
    
public:
    DebugDrawer(Shader* lineShader_)
        :
        lineShader(lineShader_) {
    }
    
    bool init();
    // debug mode functions
    virtual void setDebugMode(int debugMode_) override {
        debugMode = debugMode_;
    }
    virtual int getDebugMode() const override {
        return debugMode;
    }

    // drawing functions
    virtual void drawContactPoint(const btVector3 &pointOnB,
                                  const btVector3 &normalOnB,
                                  btScalar distance,
                                  int lifeTime,
                                  const btVector3 &color) override;
    virtual void drawLine(const btVector3 &from,
                          const btVector3 &to,
                          const btVector3 &color) override;

    // unused
    virtual void reportErrorWarning(const char* warningString) override {}
    virtual void draw3dText(const btVector3 &location, const char* textString) override {}
    
    void toggleDebugFlag(int flag);
    void prepare(RenderingContext& context);
};

#endif
