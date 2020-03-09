#include "ShadowDiffuseShader.h"

#include "Matrix4f.h"
#include "Matrix3f.h"
#include "Vector3f.h"
#include "RenderingContext.h"


static const char* vertShaderSrc =
    "#version 330 core\n"
    "layout(location = 0) in vec3 vertexPosition; "
    "layout(location = 1) in vec3 vertexNormal; "
    "layout(location = 2) in vec2 vertexTexCoord; "
    " "
    "out vec2 texCoord; "
    "out vec4 shadowTexCoord; "
    "out vec4 varyColor; "
    "out vec4 shadowColor; "
    " "
    "uniform mat4 modelViewProjectionMatrix; "
    "uniform mat4 depthBiasModelViewProjectionMatrix; "
    "uniform mat3 normalMatrix; "
    "uniform vec3 invLightDir; " // Already normalized
    "uniform vec4 lightColor; "
    "uniform vec4 ambientColor; "
    "uniform float shadowColorRate; "
    " "
    "void main() "
    "{ "
    "    vec3 worldNormal = normalize(normalMatrix * vertexNormal); "   
    "    float diffuse = dot(worldNormal, normalize(invLightDir));"
    "    "
    "    varyColor = ambientColor; "
    "    shadowColor = ambientColor; "
    "    "
    "    if(diffuse > 0.0) { "
    "        vec4 temp = lightColor * diffuse; "
    "        shadowColor += temp * shadowColorRate; "
    "        varyColor += temp; "
    "    } "
    "    "
    "    texCoord = vertexTexCoord; "
    "    "
    "    gl_Position = modelViewProjectionMatrix * vec4(vertexPosition,1); "
    "    shadowTexCoord = depthBiasModelViewProjectionMatrix * vec4(vertexPosition,1); "
    "} ";

static const char* fragShaderSrc =
    "#version 330 core\n"
    " "
    "in vec2 texCoord;  "
    "in vec4 shadowTexCoord; "
    "in vec4 varyColor; "
    "in vec4 shadowColor; "
    "out vec3 color; "
    " "
    "uniform sampler2D texSampler2D; "
    "uniform sampler2DShadow shadowMap; "
    " "
    "void main() "
    "{ "
    //"  float bias = 0.005; "
    "    vec4 baseColor = texture(texSampler2D, texCoord); "
    //"    float visibility = texture( shadowMap, vec3(shadowTexCoord.xy/shadowTexCoord.w, (shadowTexCoord.z-bias)/shadowTexCoord.w) ); "
    "    float visibility = texture( shadowMap, vec3(shadowTexCoord.xy/shadowTexCoord.w, shadowTexCoord.z/shadowTexCoord.w) ); "
    "    vec4 tmpColor = shadowColor + (varyColor - shadowColor) * visibility; "
    "    color = (baseColor * tmpColor).rgb; "
    "} ";

/**
 * <!--  init():  -->
 */
bool ShadowDiffuseShader::init() {
    bool ret = Shader::load(vertShaderSrc, fragShaderSrc);
    if( !ret ) {
        return false;
    }
    
    mvpMatrixHandle    = getUniformLocation("modelViewProjectionMatrix");
    depthBiasMvpMatrixHandle = getUniformLocation("depthBiasModelViewProjectionMatrix");
    normalMatrixHandle = getUniformLocation("normalMatrix");
    invLightDirHandle  = getUniformLocation("invLightDir");
    lightColorHandle   = getUniformLocation("lightColor");
    ambientColorHandle = getUniformLocation("ambientColor");
    shadowColorRateHandle   = getUniformLocation("shadowColorRate");

    textureHandle      = getUniformLocation("texSampler2D");
    shadowMapHandle    = getUniformLocation("shadowMap");
    return true;
}

/**
 * <!--  prepare():  -->
 */
void ShadowDiffuseShader::prepare(const RenderingContext& context) const {
    const Vector3f& lightDir = context.getLightDir();
    Vector3f invLightDir(lightDir);
    invLightDir *= -1.0f;   
    glUniform3fv( invLightDirHandle, 1,
                  (const GLfloat*)invLightDir.getPointer() );

    const Vector4f& lightColor = context.getLightColor();
    const Vector4f& ambientColor = context.getAmbientColor();
    float shadowColorRate = context.getShadowColorRate();

    glUniform4fv( lightColorHandle, 1,
                  (const GLfloat*)lightColor.getPointer() );
    glUniform4fv( ambientColorHandle, 1,
                  (const GLfloat*)ambientColor.getPointer() );
    glUniform1f( shadowColorRateHandle, shadowColorRate );
}

/**
 * <!--  setup():  -->
 */
void ShadowDiffuseShader::setup(const RenderingContext& context) const {
    const Matrix4f& modelMat = context.getModelMat();
    const Matrix4f& modelViewProjectionMat = context.getModelViewProjectionMat();
    const Matrix4f& depthBiasModelViewProjectionMat =
        context.getDepthBiasModelViewProjectionMat();
    
    // Set normal matrix by removing translate part.
    Matrix3f normalMat;
    normalMat.set(modelMat);

    glUniformMatrix3fv( normalMatrixHandle, 1, GL_FALSE,
                        (const GLfloat*)normalMat.getPointer() );

    // Set model view projection matrix
    glUniformMatrix4fv( mvpMatrixHandle, 1, GL_FALSE,
                        (const GLfloat*)modelViewProjectionMat.getPointer() );

    // Set shadow depth matrix
    glUniformMatrix4fv( depthBiasMvpMatrixHandle, 1, GL_FALSE,
                        (const GLfloat*)depthBiasModelViewProjectionMat.getPointer() );

    glUniform1i(textureHandle, 0);
    glUniform1i(shadowMapHandle, 1);
}
