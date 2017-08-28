// -*- C++ -*-
#ifndef SHADER_HEADER
#define SHADER_HEADER

#include "glinc.h"

class Vector4f;
class Vector3f;
class RenderingContext;


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
	virtual void prepare(const RenderingContext& context) const;
	virtual void setup(const RenderingContext& context) const;
	virtual void beginRender(const float* vertices) const;
	virtual void render(const unsigned short* indices, int indicesSize) const; 
	virtual void endRender() const;

	// TODO: LineShaderでしか使っていないので、共通化やめる？
	virtual void setColor(const Vector4f& color) const;
};

#endif
