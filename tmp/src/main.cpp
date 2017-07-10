#include <GLFW/glfw3.h>

#include <stdlib.h>
#include <stdio.h>

#include "Camera.h"
#include "CheckShader.h"
#include "Matrix3f.h"


/*

    [y]
     |  
     |
     |
     *------[x]
    /
   / 
 [z]
*/

static float vertices[] = {	
	// 前面
	-1.0f, -1.0f,  1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, // 左下
	 1.0f, -1.0f,  1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, // 右下
	 1.0f,  1.0f,  1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, // 右上
	-1.0f,  1.0f,  1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // 左上
  
	// 背面
	-1.0f, -1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f, // 左下
	-1.0f,  1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 1.0f, 1.0f, // 右下
	 1.0f,  1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 1.0f, 0.0f, // 右上
	 1.0f, -1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, // 左上
  
	// 上面
	-1.0f,  1.0f, -1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, // 左下
	-1.0f,  1.0f,  1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, // 右下
	 1.0f,  1.0f,  1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, // 右上
	 1.0f,  1.0f, -1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, // 左上
  
	// 底面
	-1.0f, -1.0f, -1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 1.0f, // 左下
	 1.0f, -1.0f, -1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 1.0f, // 右下
	 1.0f, -1.0f,  1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f, // 右上
	-1.0f, -1.0f,  1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, // 左上
  
	// 右側面
	 1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, // 左下
	 1.0f,  1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, // 右下
	 1.0f,  1.0f,  1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, // 右上
	 1.0f, -1.0f,  1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, // 左上
  
	// 左側面
	-1.0f, -1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, // 左下
	-1.0f, -1.0f,  1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 1.0f, // 右下
	-1.0f,  1.0f,  1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f, // 右上
	-1.0f,  1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, // 左上
};

static int verticesSize = 192;

static short indices[] = {
	0,  1,  2,      0,  2,  3,    // 前面
	4,  5,  6,      4,  6,  7,    // 背面
	8,  9,  10,     8,  10, 11,   // 上面
	12, 13, 14,     12, 14, 15,   // 底面
	16, 17, 18,     16, 18, 19,   // 右側面
	20, 21, 22,     20, 22, 23    // 左側面
};

/*
static short indices[] = {	
	0,  2,  1,      0,  3,  2,    // 前面
	4,  6,  5,      4,  7,  6,    // 背面
	8,  10, 9,      8,  11, 10,   // 上面
	12, 14, 13,     12, 15, 14,   // 底面
	16, 18, 17,     16, 19, 18,   // 右側面
	20, 22, 21,     20, 23, 22    // 左側面
};
*/

static int indicesSize = 36;

static void* readFile(const char* path, int& readSize) {
	readSize = 0;
	
	FILE* file = fopen(path, "rb");
	if ( file == nullptr ) {
		printf("Couldn't open file: %s\n", path);
		return nullptr;
	}

	int pos = ftell(file);
	fseek(file, 0, SEEK_END);
	
	int size = ftell(file);
	fseek(file, pos, SEEK_SET);

	void* buffer = malloc(size);
	int ret = fread(buffer, 1, size, file);
	if( ret != size ) {
		fclose(file);
		free(buffer);
		return nullptr;
	}

	readSize = size;
	
	fclose(file);
	return buffer;
}

#include "Image.h"
#include "PNGDecoder.h"
#include "Texture.h"

static void checkPNGDecode() {
	int size;
	void* buffer = readFile("image.png", size);
	Image image;
	PNGDecoder::decode((unsigned char*)buffer, size, image);

	Texture texture;
	texture.init((const unsigned char*)image.getBuffer(),
				 image.getWidth(), image.getHeight(),
				 image.hasAlpha());
}

// 時計回りが正面
static void error_callback(int error, const char* description) {
	fprintf(stderr, "Error: %s\n", description);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	}
}

int main() {
	GLFWwindow* window;

	glfwSetErrorCallback(error_callback);

	if (!glfwInit()) {
		exit(EXIT_FAILURE);
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

	window = glfwCreateWindow(640, 480, "Simple example", NULL, NULL);
	if (!window) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwSetKeyCallback(window, key_callback);

	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	//..
	checkPNGDecode();
	//..

	CheckShader shader;
	shader.init();
	Camera camera;

	// NOTE: OpenGL error checks have been omitted for brevity

	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	float ratio = width / (float) height;
	camera.init(1.0f, 1000.0f, 50.0f, ratio);

	//..
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
	glDisable(GL_BLEND); // 現在ObjはBLEND無し
	//..

	float head = 0.0f;
	
	while (!glfwWindowShouldClose(window)) {
		glViewport(0, 0, width, height);
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

		camera.update();

		Matrix4f mat;

		Vector4f pos(0.0f, -3.0f, -10.0f, 1.0f);

		head += 0.01f;

		mat.setIdentity();
		mat.setRotationY(head);
		mat.setColumn(3, pos);

		const Matrix4f& cameraInvMat = camera.getInvMat();
		const Matrix4f& projectionMat = camera.getProjectionMat();

		Matrix4f modelViewMat;
		modelViewMat.mul(cameraInvMat, mat);

		Matrix3f normalMat;
		normalMat.set(modelViewMat);

		Matrix4f modelViewProjectionMat;
		modelViewProjectionMat.mul(projectionMat, modelViewMat);

		shader.use();
		shader.beginRender(vertices);
		
		//glActiveTexture(GL_TEXTURE0);
	
		shader.setMatrix(modelViewProjectionMat);
		shader.setNormalMatrix(normalMat);
	
		shader.render(indices, indicesSize);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwDestroyWindow(window);

	glfwTerminate();
	
	exit(EXIT_SUCCESS);
}
