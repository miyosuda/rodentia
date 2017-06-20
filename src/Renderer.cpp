#include "Renderer.h"
#include <GLUT/glut.h>

/**
 * <!--  setProjection():  -->
 */
void Renderer::setProjection(float width, float height) {
	float aspect = width / height;
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glFrustum(-0.5f*aspect * height * 0.001f, 
			   0.5f*aspect * height * 0.001f,
			  -0.5f	       * height * 0.001f,
			   0.5f	       * height * 0.001f,
			  512.0f * 0.001f,
			  120000.0f * 0.001f);
}

/**
 * <!--  drawLine():  -->
 */
void Renderer::drawLine(const Vector4f& pos0, const Vector4f& pos1) {
	glVertex3f(pos0.x, pos0.y, pos0.z);
	glVertex3f(pos1.x, pos1.y, pos1.z);
}

/**
 * <!--  drawFloor():  -->
 */
void Renderer::drawFloor() {
	const float FIELD_MAXX = 16.0f;
	const float FIELD_MAXZ = 16.0f;

	const Vector4f p0(-FIELD_MAXX, 0.0f, -FIELD_MAXZ, 1.0f);
	const Vector4f p1(-FIELD_MAXX, 0.0f,  FIELD_MAXZ, 1.0f);
	const Vector4f p2( FIELD_MAXX, 0.0f,  FIELD_MAXZ, 1.0f);
	const Vector4f p3( FIELD_MAXX, 0.0f, -FIELD_MAXZ, 1.0f);
	
	glBegin(GL_LINES);

	glColor3f(0.0f, 0.0f, 0.0f);
	drawLine(p0, p1);
	drawLine(p1, p2);
	drawLine(p2, p3);
	drawLine(p3, p0);

	drawLine( Vector4f(-FIELD_MAXX, 0.0f, 0.0f, 1.0f),
			  Vector4f( FIELD_MAXX, 0.0f, 0.0f, 1.0f) );

	glColor3f(1.0f, 0.0f, 0.0f);
	
	drawLine( Vector4f(0.0f, 0.0f,  0.0f,       1.0f),
			  Vector4f(0.0f, 0.0f,  FIELD_MAXZ, 1.0f) );
	
	glEnd();
}

/**
 * <!--  setCamera():  -->
 */
void Renderer::setCamera(const Matrix4f& mat) {
	camera.set(mat);
}

/**
 * <!--  renderPre():  -->
 */
void Renderer::renderPre() {
	Matrix4f worldCamera;
	worldCamera.invertRT(camera);

	glMatrixMode(GL_MODELVIEW);	
	glLoadMatrixf(worldCamera.getPointer());
	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}
