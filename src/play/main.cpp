#include <stdio.h>
#include <stdlib.h>

#include "glinc.h"
#include <GLFW/glfw3.h>
#include "play.h"


#define DEFAULT_SCREEN_WIDTH   640
#define DEFAULT_SCREEN_HEIGHT  480

static int curButton = -1;

/**
 * <!--  init():  -->
 */
static void init(int width, int height) {
	playInit(width, height);
}

/**
 * release():
 */
static void release() {
	playRelease();
}

/**
 * <!--  draw():  -->
 */
static void draw(GLFWwindow* window) {
	playStep();

	glfwSwapBuffers(window);
}

/**
 * <!--  keyCallback():  -->
 */
static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if( action != GLFW_PRESS && action != GLFW_RELEASE ) {
		return;
	}

	bool press = action == GLFW_PRESS;

	int actionKey = -1;
	
	switch (key) {
	case GLFW_KEY_ESCAPE:
		if( action == GLFW_PRESS ) {
			glfwSetWindowShouldClose(window, GLFW_TRUE);
		}
		break;
	case GLFW_KEY_Q:
		actionKey = KEY_ACTION_LOOK_LEFT;
		break;
	case GLFW_KEY_E:
		actionKey = KEY_ACTION_LOOK_RIGHT;
		break;
	case GLFW_KEY_A:
		actionKey = KEY_ACTION_STRAFE_LEFT;
		break;
	case GLFW_KEY_D:
		actionKey = KEY_ACTION_STRAFE_RIGHT;
		break;
	case GLFW_KEY_W:
		actionKey = KEY_ACTION_MOVE_FORWARD;
		break;
	case GLFW_KEY_S:
		actionKey = KEY_ACTION_MOVE_BACKWARD;
		break;		
	default:
		actionKey = -1;
		break;
	}
	
	if( actionKey != -1 ) {
		playKey(actionKey, press);
	}
}

/**
 * <!--  mouseButtonCallback():  -->
 */
static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	if (button == GLFW_MOUSE_BUTTON_LEFT ||
		button == GLFW_MOUSE_BUTTON_RIGHT) {

		// MEMO: この座標はwindowのサイズベース(RetinaだとFrameBufferサイズの半分)でくる.
		double cursorX, cursorY;
		glfwGetCursorPos(window, &cursorX, &cursorY);
		int x = (int)cursorX;
		int y = (int)cursorY;

		int playButton = -1;
		if(button == GLFW_MOUSE_BUTTON_LEFT) {
			playButton = MOUSE_LEFT_BUTTON;
		} else if(button == GLFW_MOUSE_BUTTON_RIGHT) {
			playButton = MOUSE_RIGHT_BUTTON;
		}
		
		if(action == GLFW_PRESS) {
			playMouseDown(x, y, playButton);
			curButton = playButton;
		} else if( action == GLFW_RELEASE ) {
			curButton = -1;
		}
	}
}

/**
 * <!--  cursorPositionCallback():  -->
 */
static void cursorPositionCallback(GLFWwindow* window, double x, double y) {
	if( curButton != -1 ) {
		playMouseDrag((int)x, (int)y, curButton);
	}
}

/**
 * <!--  scrollCallback():  -->
 */
static void scrollCallback(GLFWwindow* window, double x, double y) {
}

/**
 * <!--  errorCallback():  -->
 */
static void errorCallback(int error, const char* description) {
	fprintf(stderr, "Error: %s\n", description);
}

/**
 * main(): 
 */
int main(int argc, char** argv) {	
	glfwSetErrorCallback(errorCallback);

	if (!glfwInit()) {
		exit(EXIT_FAILURE);
	}

	// Set GL version
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// RGB 24bit format (No alpha)
	glfwWindowHint(GLFW_RED_BITS, 8);
	glfwWindowHint(GLFW_GREEN_BITS, 8);
	glfwWindowHint(GLFW_BLUE_BITS, 8);
	
	// TODO: Tone jump occurs when explicitly set zero alpha bits.
	//glfwWindowHint(GLFW_ALPHA_BITS, 0);

	// Disable resizing
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	GLFWwindow* window = glfwCreateWindow(DEFAULT_SCREEN_WIDTH,
										  DEFAULT_SCREEN_HEIGHT,
										  "rodent", NULL, NULL);
	
	if (!window) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwSetKeyCallback(window, keyCallback);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);
	glfwSetCursorPosCallback(window, cursorPositionCallback);
	glfwSetScrollCallback(window, scrollCallback);

	glfwMakeContextCurrent(window);

	if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
		printf("Failed to init glad.\n");
		return -1;
	}
	
	glfwSwapInterval(1);

	int frameBufferWidth, frameBufferHeight;
	glfwGetFramebufferSize(window, &frameBufferWidth, &frameBufferHeight);
	
	init(frameBufferWidth, frameBufferHeight);

	while (!glfwWindowShouldClose(window)) {
		draw(window);
		
		glfwPollEvents();
	}

	release();

	exit(EXIT_SUCCESS);	
}
