// -*- C++ -*-
#ifndef SHADERMANAGER_HEADER
#define SHADERMANAGER_HEADER

#include <map>
#include <string>

using namespace std;

class Shader;

class ShaderManager {
private:	
	map<string, Shader*> shaderMap;

public:
	~ShaderManager();
	Shader* getShader(const char* name);
};

#endif
