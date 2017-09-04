#include "ShaderManager.h"

#include <stdlib.h>

#include "Shader.h"
#include "DiffuseShader.h"
#include "LineShader.h"

/**
 * <!--  ShaderManager():  -->
 */
ShaderManager::ShaderManager()
	:
	diffuseShader(nullptr),
	lineShader(nullptr) {
}

/**
 * <!--  ~ShaderManager():  -->
 */
ShaderManager::~ShaderManager() {
	release();
}

/**
 * <!--  release():  -->
 */
void ShaderManager::release() {
	if( diffuseShader != nullptr ) {
		delete diffuseShader;
		diffuseShader = nullptr;
	}
	if( lineShader != nullptr ) {
		delete lineShader;
		lineShader = nullptr;
	}	
}

/**
 * <!--  createShader():  -->
 */
Shader* ShaderManager::createShader(ShaderType shaderType) {
	Shader* shader = nullptr;
	
	switch(shaderType) {
	case DIFFUSE:
		shader = new DiffuseShader();
		break;
	case LINE:
		shader = new LineShader();
		break;
	default:
		return nullptr;
	}

	bool ret = shader->init();
	if(!ret) {
		delete shader;
		return nullptr;
	} else {
		return shader;
	}
}

/**
 * <!--  getDiffuseShader():  -->
 */
Shader* ShaderManager::getDiffuseShader() {
	if( diffuseShader == nullptr ) {
		diffuseShader = createShader(DIFFUSE);
	}
	return diffuseShader;
}

/**
 * <!--  getLineShader():  -->
 */
Shader* ShaderManager::getLineShader() {
	if( lineShader == nullptr ) {
		diffuseShader = createShader(LINE);
	}
	return lineShader;
}
