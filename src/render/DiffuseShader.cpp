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
	"    vec3 normal = normalize(normalMatrix * vertexNormal); "
	"    vec3 lightDir = vec3(1.0, -0.4, 0.3); "
	"    vec4 diffuseColor = vec4(1.0, 1.0, 1.0, 1.0); "
	"    vec4 ambientColor = vec4(0.3, 0.3, 0.3, 1.0); "
	"    "
	"    float nDotL = max(0.0, dot(normal, normalize(-lightDir)));"
	"    texCoord = vertexTexCoord; "
	"    varyColor = diffuseColor * nDotL + ambientColor; "
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
void DiffuseShader::setMatrix(const Matrix4f& modelMat,
							  const Matrix4f& modelViewMat,
							  const Matrix4f& modelViewProjectionMat) const {
	
	// Set normal matrix by removing translate part.
	Matrix3f normalMat;
	normalMat.set(modelMat);

	glUniformMatrix3fv( normalMatrixHandle, 1, GL_FALSE,
						(GLfloat*)normalMat.getPointer() );

	// Set model view projection matrix
	glUniformMatrix4fv( mvpMatrixHandle, 1, GL_FALSE,
						(GLfloat*)modelViewProjectionMat.getPointer() );
}

/**
 * <!--  beginRender():  -->
 */
void DiffuseShader::beginRender(const float* vertices) const {
	const float* normals = vertices;
	normals += 3;

	const float* texCoords = vertices;
	texCoords += 6;

	glBindBuffer(GL_ARRAY_BUFFER, 0);

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
void DiffuseShader::render(const unsigned short* indices, int indicesSize) const {
	glDrawElements(GL_TRIANGLES, indicesSize, GL_UNSIGNED_SHORT, indices);
}

/**
 * <!--  endRender():  -->
 */
void DiffuseShader::endRender() const {
	glDisableVertexAttribArray(vertexHandle);
	glDisableVertexAttribArray(normalHandle);
	glDisableVertexAttribArray(textureCoordHandle);
}
