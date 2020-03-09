// -*- C++ -*-
#ifndef SHADERMANAGER_HEADER
#define SHADERMANAGER_HEADER

#include <map>
#include <string>

using namespace std;

class Shader;

class ShaderManager {
private:
    enum ShaderType {
        DIFFUSE,
        LINE,
        SHADOW_DIFFUSE,
        SHADOW_DEPTH,
    };
    
    Shader* diffuseShader;  
    Shader* lineShader;
    Shader* shadowDiffuseShader;
    Shader* shadowDepthShader;

    Shader* createShader(ShaderType shaderType);

public:
    ShaderManager();
    ~ShaderManager();
    void release();
    Shader* getDiffuseShader(bool useShadow=true);
    Shader* getLineShader();
    Shader* getShadowDepthShader(bool useShadow=true);
};

#endif
