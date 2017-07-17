// -*- C++ -*-
#ifndef SHADER_HEADER
#define SHADER_HEADER

#include "glinc.h"

class Matrix4f;
class Vector4f;

class Shader {
protected:
	GLuint program;
	virtual void bindAttributes();
	bool load(const char* vertShaderSrc, const char* fragShaderSrc);	
	
public:
	Shader();
	virtual ~Shader();
	int compileShader(GLenum type, const char* src);

	void use();
	void release();
	
	int getUniformLocation(const char* name);
	int getAttribLocation(const char* name);

	virtual bool init()=0;
	virtual void setMatrix(const Matrix4f& modelMat,
						   const Matrix4f& modelViewMat,
						   const Matrix4f& modelViewProjectionMat) const;
	virtual void beginRender(const float* vertices) const;
	virtual void render(const short* indices, int indicesSize) const; 
	virtual void endRender() const;
	virtual void setColor(const Vector4f& color) const;
};

#endif
