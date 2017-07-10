#include "Shader.h"
#include <stdio.h>
#include <stdlib.h>

#define DEBUG //..

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
}

/**
 * <!--  bindAttributes():  -->
 */
void Shader::bindAttributes() {
	// 自前でattribute indexをbindしたい場合はオーバーライド
}

/**
 * <!--	 compileShader():  -->
 */
int Shader::compileShader(GLenum type, const char* src) {
	GLuint shader = glCreateShader(type);
	glShaderSource(shader, 1, &src, NULL);
	glCompileShader(shader);
	
#if defined(DEBUG)
	GLint logLength;
	glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
	if (logLength > 0) {
		char *log = (char*)malloc(logLength);
		glGetShaderInfoLog(shader, logLength, &logLength, log);
		printf("Shader compile log:\n%s\n", log);
		free(log);
	} else {
		//printf("Shader compile OK!\n");
	}
#endif
	
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
	
#if defined(DEBUG)
	GLint logLength;
	glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
	if (logLength > 0) {
		GLchar *log = (GLchar *)malloc(logLength);
		glGetProgramInfoLog(program, logLength, &logLength, log);
		free(log);
	}
#endif
	
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
