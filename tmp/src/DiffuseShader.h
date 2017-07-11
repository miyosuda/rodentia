// -*- C++ -*-
#ifndef DIFFUSESHADER_HEADER
#define DIFFUSESHADER_HEADER

#include "Shader.h"

class Matrix4f;
class Matrix3f;

class DiffuseShader : public Shader {
private:
	int vertexHandle;
	int normalHandle;
	int textureCoordHandle;
	int mvpMatrixHandle;
	int normalMatrixHandle;
	
public:
	bool init();
	void setMatrix(const Matrix4f& mat);
	void setNormalMatrix(const Matrix3f& mat);
	void beginRender(const float* vertices);
	void render(const short* indices, int indicesSize);
	void endRender();
};

#endif
