#include <stdio.h>
#include <stdlib.h>
#include <GLFW/glfw3.h>

#include "play.h"
#include "Matrix4f.h"

#define DEFAULT_SCREEN_WIDTH   640
#define DEFAULT_SCREEN_HEIGHT  480

static int curButton = -1;

/**
 * setProjection():
 */
static void setProjection(float width, float height) {
	float aspect = (float)DEFAULT_SCREEN_WIDTH / (float)DEFAULT_SCREEN_HEIGHT;
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glFrustum(-0.5f*aspect * DEFAULT_SCREEN_HEIGHT * 0.001f, 
			   0.5f*aspect * DEFAULT_SCREEN_HEIGHT * 0.001f,
			  -0.5f	       * DEFAULT_SCREEN_HEIGHT * 0.001f,
			   0.5f	       * DEFAULT_SCREEN_HEIGHT * 0.001f,
			  512.0f * 0.001f,
			  120000.0f * 0.001f);

	glMatrixMode(GL_MODELVIEW);
}

/**
 * <!--  init():  -->
 */
static void init() {
	glDepthFunc(GL_LESS);
	glEnable(GL_DEPTH_TEST);
	
	glClearColor(0.5f, 0.5f, 0.5f, 1.0f);

	playInit();
}

/**
 * release():
 */
static void release() {
	playFinalize();
}

/**
 * <!--  draw():  -->
 */
static void draw(GLFWwindow* window) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	playLoop();	

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
 * <!--  framebufferSizeCallback():  -->
 */
static void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
	setProjection(width, height);

	playReshape(width, height);
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
	GLFWwindow* window;

	glfwSetErrorCallback(errorCallback);

	if (!glfwInit()) {
		exit(EXIT_FAILURE);
	}

	window = glfwCreateWindow(640, 480, "Wave Simulation", NULL, NULL);
	if (!window) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwSetKeyCallback(window, keyCallback);
	glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);
	glfwSetCursorPosCallback(window, cursorPositionCallback);
	glfwSetScrollCallback(window, scrollCallback);

	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	framebufferSizeCallback(window, width, height);

	init();

	while (!glfwWindowShouldClose(window)) {
		draw(window);
		
		glfwPollEvents();
	}

	release();

	exit(EXIT_SUCCESS);	
}
