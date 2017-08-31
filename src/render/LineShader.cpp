#include "LineShader.h"

#include "Matrix4f.h"
#include "RenderingContext.h"


static const char* vertShaderSrc =
	"#version 330\n"
	"layout(location = 0) in vec4 vertexPosition; "
	""
	"out vec4 varyColor; "
	""
	"uniform mat4 modelViewProjectionMatrix; "
	"uniform vec4 lineColor; "
	"" 	
	"void main() "
	"{ "
	"    varyColor = lineColor; "
	"    gl_Position = modelViewProjectionMatrix * vertexPosition; "
	"} ";

static const char* fragShaderSrc =
	"#version 330\n"
	" "
	"in vec4 varyColor; "
	"out vec3 color; "	
	" "
	"void main() "
	"{ "
	"    color = varyColor; "
	"} ";

/**
 * <!--  init():  -->
 */
bool LineShader::init() {
	bool ret = Shader::load(vertShaderSrc, fragShaderSrc);
	if( !ret ) {
		return false;
	}
	
	vertexHandle    = getAttribLocation("vertexPosition");
	mvpMatrixHandle = getUniformLocation("modelViewProjectionMatrix");
	lineColorHandle = getUniformLocation("lineColor");

	return true;
}

/**
 * <!--  setup():  -->
 */
void LineShader::setup(const RenderingContext& context) const {
	const Matrix4f& modelViewProjectionMat = context.getModelViewProjectionMat();
	
	// Set model view projection matrix
	glUniformMatrix4fv( mvpMatrixHandle, 1, GL_FALSE,
						(const GLfloat*)modelViewProjectionMat.getPointer() );
}

/**
 * <!--  beginRender():  -->
 */
void LineShader::beginRender(const float* vertices) const {
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	// Now assuming 3 floats packed vertices
	glVertexAttribPointer(vertexHandle, 3, GL_FLOAT, GL_FALSE,
						  0, vertices);
	glEnableVertexAttribArray(vertexHandle);
}

/**
 * <!--  render():  -->
 */
void LineShader::render(const unsigned short* indices, int indicesSize) const {
	glDrawElements(GL_LINES, indicesSize, GL_UNSIGNED_SHORT, indices);
}

/**
 * <!--  endRender():  -->
 */
void LineShader::endRender() const {
	glDisableVertexAttribArray(vertexHandle);
}

/**
 * <!--  setColor():  -->
 */
void LineShader::setColor(const Vector4f& color) const {
	glUniform4fv( lineColorHandle, 1, (const GLfloat*)color.getPointer() );
}
