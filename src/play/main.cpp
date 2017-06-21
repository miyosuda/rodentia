#include <stdio.h>
#include <stdlib.h>

#include <GLFW/glfw3.h>

#include "play.h"
#include "Matrix4f.h"

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
	if (action != GLFW_PRESS) {
		return;
	}

	switch (key) {
		case GLFW_KEY_ESCAPE:
			glfwSetWindowShouldClose(window, GLFW_TRUE);
			break;
		default:
			break;
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
