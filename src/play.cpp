#include "play.h"
#include <assert.h>
#include <GLUT/glut.h>

#include "TrackBall.h"
#include "Environment.h"

#include "ScreenRenderer.h" //..

static TrackBall trackBall(0.0f, 0.0f, 8.0f, -0.3f);
static Environment environment;
static ScreenRenderer renderer; //..

/**
 * reshape():
 */
void playReshape(int width, int height) {
	renderer.init(width, height); //..
	
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
	environment.init();
}

/**
 * playLoop():
 */
void playLoop() {
	Matrix4f mat;
	trackBall.getMat(mat);
	renderer.setCamera(mat);
	
	renderer.renderPre();
	
	environment.step();
	
	renderer.render();
}

/**
 * playFinalize():
 */
void playFinalize() {
	environment.release();
}
