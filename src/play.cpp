#include "play.h"
#include <assert.h>
#include <GLUT/glut.h>

#include "TrackBall.h"
#include "rigid.h"

static TrackBall trackBall(0.0f, 0.0f, 8.0f, -0.3f);
static RigidManager rigidManager; //..  TODO:

/**
 * reshape():
 */
void playReshape(int width, int height) {
	trackBall.resize(width, height);
}

/**
 * mouseDown():
 */
void playMouseDown(int x, int y, int button) {
	if(button == MOUSE_LEFT_BUTTON) {
		trackBall.startRotation(x, y);
	} else if(button == MOUSE_RIGHT_BUTTON) {
		trackBall.startZoom(x, y);
	}
}

/**
 * mouseDrag():
 */
void playMouseDrag(int x, int y, int button) {
	if(button == MOUSE_LEFT_BUTTON) {
		trackBall.dragRotation(x, y);
	} else if(button == MOUSE_RIGHT_BUTTON) {
		trackBall.dragZoom(x, y);
	}
}

/**
 * playInit():
 */
void playInit() {
	rigidManager.initPhysics();
}

static void drawLine(const Vector4f& pos0, const Vector4f& pos1) {
	glVertex3f(pos0.x, pos0.y, pos0.z);
	glVertex3f(pos1.x, pos1.y, pos1.z);
}

static void drawFloor() {
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
	
	drawLine( Vector4f(0.0f, 0.0f, -FIELD_MAXZ, 1.0f),
			  Vector4f(0.0f, 0.0f,  FIELD_MAXZ, 1.0f) );
	
	glEnd();
}

/**
 * playLoop():
 */
void playLoop() {
	Matrix4f camera;
	trackBall.getMat(camera);

	Matrix4f worldCamera;
	worldCamera.invertRT(camera);

	glPushMatrix();
	{
		glMultMatrixf( (const float*)&worldCamera );
		
		rigidManager.stepSimulation(1.0f/60.0f); //..
		
		drawFloor();
	}
	glPopMatrix();	
}

/**
 * playFinalize():
 */
void playFinalize() {
	rigidManager.exitPhysics();
}
