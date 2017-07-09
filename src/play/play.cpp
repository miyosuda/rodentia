#include "play.h"
#include <assert.h>
#include <GLFW/glfw3.h>

#include "TrackBall.h"
#include "Environment.h"

static TrackBall trackBall(0.0f, 0.0f, 8.0f, -0.3f);
static Environment environment;

static bool lookLeftState     = false;
static bool lookRightState    = false;
static bool strafeLeftState   = false;
static bool strafeRightState  = false;
static bool moveForwardState  = false;
static bool moveBackwardState = false;

/**
 * <!--  playMouseDown():  -->
 */
void playMouseDown(int x, int y, int button) {
	if(button == MOUSE_LEFT_BUTTON) {
		trackBall.startRotation(x, y);
	} else if(button == MOUSE_RIGHT_BUTTON) {
		trackBall.startZoom(x, y);
	}
}

/**
 * <!--  playMouseDrag():  -->
 */
void playMouseDrag(int x, int y, int button) {
	if(button == MOUSE_LEFT_BUTTON) {
		trackBall.dragRotation(x, y);
	} else if(button == MOUSE_RIGHT_BUTTON) {
		trackBall.dragZoom(x, y);
	}
}

/**
 * <!--  playKey():  -->
 */
void playKey(int actionKey, bool press) {
	switch(actionKey) {
	case KEY_ACTION_LOOK_LEFT:
		lookLeftState = press;
		break;
	case KEY_ACTION_LOOK_RIGHT:
		lookRightState = press;
		break;
	case KEY_ACTION_STRAFE_LEFT:
		strafeLeftState = press;
		break;
	case KEY_ACTION_STRAFE_RIGHT:
		strafeRightState = press;
		break;
	case KEY_ACTION_MOVE_FORWARD:
		moveForwardState = press;
		break;		 
	case KEY_ACTION_MOVE_BACKWARD:
		moveBackwardState = press;
		break;
	default:
		break;
	}
}

/**
 * <!--  playInit():  -->
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

	// Locate test objects
	environment.addBox(3.0f, 3.0f, 3.0f,
					   10.0f, 3.0f, 3.0f,
					   3.141592f * 0.25f,
					   false);
	environment.addSphere(1.0f,
						  5.0f, 1.0f, -5.0f,
						  0.0f,
						  true);
}

static void getAction(Action& action) {
	int lookAction = 0;
	int strafeAction = 0;
	int moveAction = 0;

	if( lookLeftState ) {
		lookAction += 10;
	}
	if( lookRightState ) {
		lookAction -= 10;
	}
	if( strafeLeftState ) {
		strafeAction += 1;
	}
	if(strafeRightState ) {
		strafeAction -= 1;
	}
	if( moveForwardState ) {
		moveAction += 1;
	}
	if( moveBackwardState ) {
		moveAction -= 1;
	}

	action.set(lookAction, strafeAction, moveAction);
}

/**
 * <!--  playStep():  -->
 */
void playStep() {
	Matrix4f mat;
	trackBall.getMat(mat);
	environment.setRenderCamera(mat);

	Action action;
	getAction(action);
	
	environment.step(action);
}

/**
 * <!--  playRelease():  -->
 */
void playRelease() {
	environment.release();
}
