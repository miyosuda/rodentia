#include "play.h"
#include <assert.h>

#include "Graphics.h"
#include "TrackBall.h"

#include "rigid.h"

static TrackBall trackBall(0.0f, 0.0f, -8.0f, -0.3f);
static RigidManager rigidManager; //..

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
	if(button == MOUSE_LELFT_BUTTON) {
		trackBall.startRotation(x, y);
	} else if(button == MOUSE_RIGHT_BUTTON) {
		trackBall.startZoom(x, y);
	}
}

/**
 * mouseDrag():
 */
void playMouseDrag(int x, int y, int button) {
	if(button == MOUSE_LELFT_BUTTON) {
		trackBall.dragRotation(x, y);
	} else if(button == MOUSE_RIGHT_BUTTON) {
		trackBall.dragZoom(x, y);
	}
}

/**
 * playInit():
 */
void playInit() {
	Graphics::getGraphics().init();

	rigidManager.initPhysics();
}

/**
 * playLoop():
 */
void playLoop() {
	rigidManager.stepSimulation(1.0f/60.0f); //..
	
	Matrix4f camera;
	trackBall.getMat(camera);

	Graphics::getGraphics().setCamera( camera );

	//..
#define FIELD_MAXX 16.0f
#define FIELD_MAXZ 16.0f
	const Vector4f black(0.0f, 0.0f, 0.0f, 0.0f);
	const Vector4f red(1.0f, 0.0f, 0.0f, 0.0f);

	const Vector4f p0(-FIELD_MAXX, 0.0f, -FIELD_MAXZ, 1.0f);
	const Vector4f p1(-FIELD_MAXX, 0.0f,  FIELD_MAXZ, 1.0f);
	const Vector4f p2( FIELD_MAXX, 0.0f,  FIELD_MAXZ, 1.0f);
	const Vector4f p3( FIELD_MAXX, 0.0f, -FIELD_MAXZ, 1.0f);

	Graphics& g = Graphics::getGraphics();

	g.drawLine(p0, p1, black);
	g.drawLine(p1, p2, black);
	g.drawLine(p2, p3, black);
	g.drawLine(p3, p0, black);

	g.drawLine( Vector4f(-FIELD_MAXX, 0.0f, 0.0f, 1.0f),
				Vector4f( FIELD_MAXX, 0.0f, 0.0f, 1.0f),
				black );

	g.drawLine( Vector4f(0.0f, 0.0f, -FIELD_MAXZ, 1.0f),
				Vector4f(0.0f, 0.0f,  FIELD_MAXZ, 1.0f),
				red );
	//..
}

/**
 * playFinalize():
 */
void playFinalize() {
	rigidManager.exitPhysics();
}
