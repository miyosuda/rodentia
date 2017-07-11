#include "DiffuseShader.h"

#include "Matrix4f.h"
#include "Matrix3f.h"

static const char* vertShaderSrc =
	"#version 110\n"
	"attribute vec4 vertexPosition; "
	"attribute vec3 vertexNormal; "
	"attribute vec2 vertexTexCoord; "
	""
	"varying vec2 texCoord; "
	"varying vec4 varyColor; "
	""
	"uniform mat4 modelViewProjectionMatrix; "
	"uniform mat3 normalMatrix; "
	"" 	
	"void main() "
	"{ "
	"    vec3 eyeNormal = normalize(normalMatrix * vertexNormal); "
	"    vec3 lightPosition = vec3(-1.0, -0.5, -1.0); "
	"    vec4 diffuseColor = vec4(1.0, 1.0, 1.0, 1.0); "
	"    vec4 ambientColor = vec4(0.3, 0.3, 0.3, 1.0); "
	"    "
	"    float nDotVP = max(0.0, dot(eyeNormal, normalize(lightPosition)));"
	"    texCoord = vertexTexCoord; "
	"    varyColor = diffuseColor * nDotVP + ambientColor; "
	"    varyColor.w = 1.0;	"
	"    "
	"    gl_Position = modelViewProjectionMatrix * vertexPosition; "
	"} ";

static const char* fragShaderSrc =
	"#version 110\n"
	" "
	"varying vec2 texCoord;	"
	"varying vec4 varyColor; "
	" "
	"uniform sampler2D texSampler2D; "
	" "
	"void main() "
	"{ "
	"    vec4 baseColor = texture2D(texSampler2D, texCoord); " 	
	"    gl_FragColor = baseColor * varyColor; "
	"} ";

/**
 * <!--  init():  -->
 */
bool DiffuseShader::init() {
	bool ret = Shader::load(vertShaderSrc, fragShaderSrc);
	if( !ret ) {
		return false;
	}
	
	vertexHandle       = getAttribLocation("vertexPosition");
	normalHandle       = getAttribLocation("vertexNormal");
	textureCoordHandle = getAttribLocation("vertexTexCoord");
	mvpMatrixHandle    = getUniformLocation("modelViewProjectionMatrix");
	normalMatrixHandle = getUniformLocation("normalMatrix");

	return true;
}

/**
 * <!--  setMatrix():  -->
 */
void DiffuseShader::setMatrix(const Matrix4f& mat) {
	glUniformMatrix4fv( mvpMatrixHandle, 1, GL_FALSE,
						(GLfloat*)mat.getPointer() );
}

/**
 * <!--  setMatrix():  -->
 */
void DiffuseShader::setNormalMatrix(const Matrix3f& mat) {
	glUniformMatrix3fv( normalMatrixHandle, 1, GL_FALSE,
						(GLfloat*)mat.getPointer() );
}

/**
 * <!--  beginRender():  -->
 */
void DiffuseShader::beginRender(const float* vertices) {
	const float* normals = vertices;
	normals += 3;

	const float* texCoords = vertices;
	texCoords += 6;

	glVertexAttribPointer(vertexHandle, 3, GL_FLOAT, GL_FALSE,
						  4*8, vertices);
	glEnableVertexAttribArray(vertexHandle);

	glVertexAttribPointer(normalHandle, 3, GL_FLOAT, GL_FALSE,
						  4*8, normals);
	glEnableVertexAttribArray(normalHandle);

	glVertexAttribPointer(textureCoordHandle, 2, GL_FLOAT, GL_FALSE,
						  4*8, texCoords);
	glEnableVertexAttribArray(textureCoordHandle);
}

/**
 * <!--  render():  -->
 */
void DiffuseShader::render(const short* indices, int indicesSize) {
	glDrawElements(GL_TRIANGLES, indicesSize, GL_UNSIGNED_SHORT, indices);
}

/**
 * <!--  endRender():  -->
 */
void DiffuseShader::endRender() {
	glDisableVertexAttribArray(vertexHandle);
	glDisableVertexAttribArray(normalHandle);
	glDisableVertexAttribArray(textureCoordHandle);
}
