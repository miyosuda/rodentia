// -*- C++ -*-
#ifndef SHADER_HEADER
#define SHADER_HEADER

#include "glinc.h"

class Shader {
protected:
	GLuint program;
	virtual void bindAttributes();
	
public:
	Shader();
	virtual ~Shader();
	int compileShader(GLenum type, const char* src);
	bool load(const char* vertShaderSrc, const char* fragShaderSrc);
	void use();
	void release();
	
	int getUniformLocation(const char* name);
	int getAttribLocation(const char* name);
};

#endif
