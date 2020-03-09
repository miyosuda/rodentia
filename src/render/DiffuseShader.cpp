#include "DiffuseShader.h"

#include "Matrix4f.h"
#include "Matrix3f.h"
#include "Vector3f.h"
#include "RenderingContext.h"


static const char* vertShaderSrc =
    "#version 330 core\n"
    "layout(location = 0) in vec4 vertexPosition; "
    "layout(location = 1) in vec3 vertexNormal; "
    "layout(location = 2) in vec2 vertexTexCoord; "
    ""
    "out vec2 texCoord; "
    "out vec4 varyColor; "
    ""
    "uniform mat4 modelViewProjectionMatrix; "
    "uniform mat3 normalMatrix; "
    "uniform vec3 invLightDir; " // Already normalized
    ""
    "void main() "
    "{ "
    "    vec3 normal = normalize(normalMatrix * vertexNormal); "
    "    vec4 diffuseColor = vec4(1.0, 1.0, 1.0, 1.0); "
    "    vec4 ambientColor = vec4(0.3, 0.3, 0.3, 1.0); "
    "    "
    "    float nDotL = max(0.0, dot(normal, invLightDir));"
    "    texCoord = vertexTexCoord; "
    "    varyColor = diffuseColor * nDotL + ambientColor; "
    "    varyColor.w = 1.0; "
    "    "
    "    gl_Position = modelViewProjectionMatrix * vertexPosition; "
    "} ";

static const char* fragShaderSrc =
    "#version 330 core\n"
    " "
    "in vec2 texCoord;  "
    "in vec4 varyColor; "
    "out vec3 color; "  
    " "
    "uniform sampler2D texSampler2D; "
    " "
    "void main() "
    "{ "
    "    vec4 baseColor = texture(texSampler2D, texCoord); "
    "    color = (baseColor * varyColor).rgb; "
    "} ";

/**
 * <!--  init():  -->
 */
bool DiffuseShader::init() {
    bool ret = Shader::load(vertShaderSrc, fragShaderSrc);
    if( !ret ) {
        return false;
    }
    
    mvpMatrixHandle    = getUniformLocation("modelViewProjectionMatrix");
    normalMatrixHandle = getUniformLocation("normalMatrix");
    invLightDirHandle  = getUniformLocation("invLightDir");
    lightColorHandle   = getUniformLocation("lightColor");
    ambientColorHandle = getUniformLocation("ambientColor");

    return true;
}

/**
 * <!--  prepare():  -->
 */
void DiffuseShader::prepare(const RenderingContext& context) const {
    use();
    
    const Vector3f& lightDir = context.getLightDir();
    Vector3f invLightDir(lightDir);
    invLightDir *= -1.0f;
    
    glUniform3fv( invLightDirHandle, 1,
                  (const GLfloat*)invLightDir.getPointer() );

    const Vector4f& lightColor = context.getLightColor();
    const Vector4f& ambientColor = context.getAmbientColor();

    glUniform4fv( lightColorHandle, 1,
                  (const GLfloat*)lightColor.getPointer() );
    glUniform4fv( ambientColorHandle, 1,
                  (const GLfloat*)ambientColor.getPointer() );
}

/**
 * <!--  setup():  -->
 */
void DiffuseShader::setup(const RenderingContext& context) const {
    const Matrix4f& modelMat = context.getModelMat();
    const Matrix4f& modelViewProjectionMat = context.getModelViewProjectionMat();
    
    // Set normal matrix by removing translate part.
    Matrix3f normalMat;
    normalMat.set(modelMat);

    glUniformMatrix3fv( normalMatrixHandle, 1, GL_FALSE,
                        (const GLfloat*)normalMat.getPointer() );

    // Set model view projection matrix
    glUniformMatrix4fv( mvpMatrixHandle, 1, GL_FALSE,
                        (const GLfloat*)modelViewProjectionMat.getPointer() );
}
