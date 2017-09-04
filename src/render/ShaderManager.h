// -*- C++ -*-
#ifndef SHADERMANAGER_HEADER
#define SHADERMANAGER_HEADER

#include <map>
#include <string>

using namespace std;

class Shader;

class ShaderManager {
private:
	enum ShaderType {
		DIFFUSE,
		LINE
	};
	
	Shader* diffuseShader;	
	Shader* lineShader;

	Shader* createShader(ShaderType shaderType);

public:
	ShaderManager();
	~ShaderManager();
	void release();
	Shader* getDiffuseShader();
	Shader* getLineShader();
};

#endif
