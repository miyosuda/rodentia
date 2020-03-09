#include "ShaderManager.h"

#include <stdlib.h>

#include "Shader.h"
#include "DiffuseShader.h"
#include "LineShader.h"
#include "ShadowDiffuseShader.h"
#include "ShadowDepthShader.h"

/**
 * <!--  ShaderManager():  -->
 */
ShaderManager::ShaderManager()
    :
    diffuseShader(nullptr),
    lineShader(nullptr),
    shadowDiffuseShader(nullptr),
    shadowDepthShader(nullptr)
{
}

/**
 * <!--  ~ShaderManager():  -->
 */
ShaderManager::~ShaderManager() {
    release();
}

/**
 * <!--  release():  -->
 */
void ShaderManager::release() {
    if( diffuseShader != nullptr ) {
        delete diffuseShader;
        diffuseShader = nullptr;
    }
    if( lineShader != nullptr ) {
        delete lineShader;
        lineShader = nullptr;
    }
    if( shadowDiffuseShader != nullptr ) {
        delete shadowDiffuseShader;
        shadowDiffuseShader = nullptr;
    }
    if( shadowDepthShader != nullptr ) {
        delete shadowDepthShader;
        shadowDepthShader = nullptr;
    }   
}

/**
 * <!--  createShader():  -->
 */
Shader* ShaderManager::createShader(ShaderType shaderType) {
    Shader* shader = nullptr;
    
    switch(shaderType) {
    case DIFFUSE:
        shader = new DiffuseShader();
        break;
    case LINE:
        shader = new LineShader();
        break;
    case SHADOW_DIFFUSE:
        shader = new ShadowDiffuseShader();
        break;
    case SHADOW_DEPTH:
        shader = new ShadowDepthShader();
        break;
    default:
        return nullptr;
    }

    bool ret = shader->init();
    if(!ret) {
        delete shader;
        return nullptr;
    } else {
        return shader;
    }
}

/**
 * <!--  getDiffuseShader():  -->
 */
Shader* ShaderManager::getDiffuseShader(bool useShadow) {
    if( useShadow ) {
        if( shadowDiffuseShader == nullptr ) {
            shadowDiffuseShader = createShader(SHADOW_DIFFUSE);
        }
        return shadowDiffuseShader;
    } else {
        if( diffuseShader == nullptr ) {
            diffuseShader = createShader(DIFFUSE);
        }
        return diffuseShader;
    }
}

/**
 * <!--  getLineShader():  -->
 */
Shader* ShaderManager::getLineShader() {
    if( lineShader == nullptr ) {
        diffuseShader = createShader(LINE);
    }
    return lineShader;
}

/**
 * <!--  getShadowDepthShader():  -->
 */
Shader* ShaderManager::getShadowDepthShader(bool useShadow) {
    if( useShadow ) {
        if( shadowDepthShader == nullptr ) {
            shadowDepthShader = createShader(SHADOW_DEPTH);
        }
        return shadowDepthShader;
    } else {
        return nullptr;
    }
}
