#include "ShadowDepthShader.h"

#include "Matrix4f.h"
#include "RenderingContext.h"

static const char* vertShaderSrc =
    "#version 330 core\n"
    //"layout(location = 0) in vec4 vertexPosition; " // TODO:
    "layout(location = 0) in vec3 vertexPosition; "
    ""
    "uniform mat4 modelViewProjectionMatrix; "
    ""
    "void main() "
    "{ "
    //"    gl_Position = modelViewProjectionMatrix * vertexPosition; " // TODO
    "    gl_Position = modelViewProjectionMatrix * vec4(vertexPosition,1); "
    "} ";

static const char* fragShaderSrc =
    "#version 330 core\n"
    "layout(location = 0) out float fragmentDepth; "
    " "
    "void main() "
    "{ "
    "    fragmentDepth = gl_FragCoord.z; "
    "} ";

/**
 * <!--  init():  -->
 */
bool ShadowDepthShader::init() {
    bool ret = Shader::load(vertShaderSrc, fragShaderSrc);
    if( !ret ) {
        return false;
    }
    
    mvpMatrixHandle = getUniformLocation("modelViewProjectionMatrix");
    return true;
}

/**
 * <!--  prepare():  -->
 */
void ShadowDepthShader::prepare(const RenderingContext& context) const {
}

/**
 * <!--  setup():  -->
 */
void ShadowDepthShader::setup(const RenderingContext& context) const {
    const Matrix4f& depthModelViewProjectionMat = 
        context.getDepthModelViewProjectionMat();
    
    // Set model view projection matrix
    glUniformMatrix4fv( mvpMatrixHandle, 1, GL_FALSE,
                        (const GLfloat*)depthModelViewProjectionMat.getPointer() );
}
