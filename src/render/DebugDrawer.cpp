#include "DebugDrawer.h"
#include "glinc.h"
#include "Shader.h"
#include "Matrix4f.h"
#include "RenderingContext.h"


/**
 * <!--  drawLine():  -->
 */
void DebugDrawer::drawLine(const btVector3 &from,
						   const btVector3 &to,
						   const btVector3 &color) {

	// draws a simple line of pixels between points.

	lineShader->setColor(Vector4f(color.x(), color.y(), color.z(), 1.0f));

	float vertices[6];
	vertices[0] = from.x();
	vertices[1] = from.y();
	vertices[2] = from.z();
	vertices[3] = to.x();
	vertices[4] = to.y();
	vertices[5] = to.z();

	lineShader->use();

	// TODO:
	/*
	lineShader->beginRender(vertices);
	unsigned short indices[2];
	indices[0] = 0;
	indices[1] = 1;
	
	lineShader->render(indices, 2);
	lineShader->endRender();
	*/
}

/**
 * <!--  drawContactPoint():  -->
 */
void DebugDrawer::drawContactPoint(const btVector3 &pointOnB,
								   const btVector3 &normalOnB,
								   btScalar distance,
								   int lifeTime,
								   const btVector3 &color) {
	// draws a line between two contact points
	btVector3 const startPoint = pointOnB;
	btVector3 const endPoint = pointOnB + normalOnB * distance;
	drawLine(startPoint, endPoint, color);
}

/**
 * <!--  toggleDebugFlag():  -->
 */
void DebugDrawer::toggleDebugFlag(int flag) {
	// checks if a flag is set and enables/
	// disables it
	if (debugMode & flag) {
		// flag is enabled, so disable it
		debugMode = debugMode & (~flag);
	} else {
		// flag is disabled, so enable it
		debugMode |= flag;
	}
}

/**
 * <!--  prepare():  -->
 */
void DebugDrawer::prepare(RenderingContext& context) {
	Matrix4f modelMat;
	modelMat.setIdentity();

	context.setModelMat(modelMat);
	lineShader->setup(context);
}
