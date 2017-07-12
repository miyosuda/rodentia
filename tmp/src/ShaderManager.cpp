#include "ShaderManager.h"

#include <stdlib.h>

#include "Shader.h"
#include "DiffuseShader.h"
#include "LineShader.h"

/**
 * <!--  ~ShaderManager():  -->
 */
ShaderManager::~ShaderManager() {
	for (auto itr=shaderMap.begin(); itr!=shaderMap.end(); ++itr) {
		Shader* shader = itr->second;
		delete shader;
	}
	shaderMap.clear();
}

/**
 * <!--  getShader():  -->
 */
Shader* ShaderManager::getShader(const char* name) {
	string nameStr = name;
	
	auto itr = shaderMap.find(nameStr);
	if( itr != shaderMap.end() ) {
		Shader* shader = itr->second;
		return shader;
	}

	Shader* shader = nullptr;

	if( nameStr == "diffuse" ) {
		shader = new DiffuseShader();
	} else if( nameStr == "line" ) {
		shader = new LineShader();
	}

	if( shader != nullptr ) {
		bool ret = shader->init();
		if( !ret ) {
			delete shader;
			return nullptr;
		}
		shaderMap[nameStr] = shader;
	}
	
	return shader;
}
