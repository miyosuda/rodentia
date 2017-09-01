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
	"    color = varyColor.rgb; "
	"} ";

/**
 * <!--  init():  -->
 */
bool LineShader::init() {
	bool ret = Shader::load(vertShaderSrc, fragShaderSrc);
	if( !ret ) {
		return false;
	}
	
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
 * <!--  setColor():  -->
 */
void LineShader::setColor(const Vector4f& color) const {
	glUniform4fv( lineColorHandle, 1, (const GLfloat*)color.getPointer() );
}
