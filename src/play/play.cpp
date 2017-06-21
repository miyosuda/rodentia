#include "play.h"
#include <assert.h>
#include <GLUT/glut.h>

#include "TrackBall.h"
#include "Environment.h"

static TrackBall trackBall(0.0f, 0.0f, 8.0f, -0.3f);
static Environment environment;

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
void playInit(int width, int height) {
	// MEMO: ここのwidth, heightはフレームバッファのサイズになっている.
	
#if defined(__APPLE__)
	// Workaround for retina MacBook Pro
	trackBall.resize(width/2, height/2);
#else
	trackBall.resize(width, height);
#endif

	environment.init();

	environment.initRenderer(width, height, false);
}

/**
 * playStep():
 */
void playStep() {
	Matrix4f mat;
	trackBall.getMat(mat);
	environment.setRenderCamera(mat);
	
	environment.step();
}

/**
 * <!--  playRelease():  -->
 */
void playRelease() {
	environment.release();
}
