#include "Shader.h"
#include <stdio.h>
#include <stdlib.h>
#include "Matrix4f.h"
#include "Vector3f.h"

/**
 * <!--	 Shader():	-->
 */
Shader::Shader() {
	program = 0;
}

/**
 * <!--	 ~Shader():	 -->
 */
Shader::~Shader() {
	release();
}

/**
 * <!--  bindAttributes():  -->
 */
void Shader::bindAttributes() {
	// Override if wan't to bind attribute index manually
}

/**
 * <!--	 compileShader():  -->
 */
int Shader::compileShader(GLenum type, const char* src) {
	GLuint shader = glCreateShader(type);
	glShaderSource(shader, 1, &src, NULL);
	glCompileShader(shader);
	
	GLint logLength;
	glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
	if (logLength > 0) {
		char *log = (char*)malloc(logLength);
		glGetShaderInfoLog(shader, logLength, &logLength, log);
		printf("Shader compile log:\n%s\n", log);
		free(log);
	}
	
	GLint status;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
	if (status == 0) {
		glDeleteShader(shader);
		return -1;
	}
	
	return (int)shader;
}

/**
 * <!--	 load():  -->
 */
bool Shader::load(const char* vertShaderSrc, const char* fragShaderSrc) {
	program = glCreateProgram();

	GLuint vertShader, fragShader;

	int vertShader_ = compileShader(GL_VERTEX_SHADER, vertShaderSrc);
	if( vertShader_ < 0 ) {
		return false;
	}
	vertShader = vertShader_;
	
	int fragShader_ = compileShader(GL_FRAGMENT_SHADER, fragShaderSrc);
	if( fragShader_ < 0 ) {
		glDeleteShader(vertShader);
		vertShader = 0;
		return false;
	}
	fragShader = fragShader_;
	
	glAttachShader(program, vertShader);
	glAttachShader(program, fragShader);
	
	bindAttributes();

	glLinkProgram(program);
	
	GLint logLength;
	glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
	if (logLength > 0) {
		GLchar *log = (GLchar *)malloc(logLength);
		glGetProgramInfoLog(program, logLength, &logLength, log);
		free(log);
	}
	
	GLint status;
	glGetProgramiv(program, GL_LINK_STATUS, &status);
	if (status == 0) {
		if (vertShader) {
			glDeleteShader(vertShader);
			vertShader = 0;
		}
		if (fragShader) {
			glDeleteShader(fragShader);
			fragShader = 0;
		}
		if (program) {
			glDeleteProgram(program);
			program = 0;
		}
		return false;
	}

	if (vertShader) {
		glDeleteShader(vertShader);
	}
	if (fragShader) {
		glDeleteShader(fragShader);
	}
	
	return true;
}

/**
 * <!--  use():  -->
 */
void Shader::use() {
	glUseProgram(program);
}

/**
 * <!--	 release():	 -->
 */
void Shader::release() {
	if (program) {
		glDeleteProgram(program);
		program = 0;
	}
}

/**
 * <!--  getUniformLocation():  -->
 */
int Shader::getUniformLocation(const char* name) {
	return glGetUniformLocation(program, name);
}

/**
 * <!--  getAttribLocation():  -->
 */
int Shader::getAttribLocation(const char* name) {
	return glGetAttribLocation(program, name);
}

/**
 * <!--  setup():  -->
 */
void Shader::setup(const RenderingContext& context) const {
}

/**
 * <!--  beginRender():  -->
 */
void Shader::beginRender(const float* vertices) const {
}

/**
 * <!--  render():  -->
 */
void Shader::render(const unsigned short* indices, int indicesSize) const {
}

/**
 * <!--  endRender():  -->
 */
void Shader::endRender() const {
}

/**
 * <!--  setColor():  -->
 */
void Shader::setColor(const Vector4f& color) const {
}

/**
 * <!--  setDirectionalLight():  -->
 */
void Shader::setDirectionalLight(const Vector3f& lightDir) const {
}
