#include "rodent_api.h"
#include <stdio.h>

#include "Environment.h"

//..
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
//..

void* rodent_create() {
	Environment* environment = new Environment();
	return static_cast<void*>(environment);
}

int rodent_init(void* context_) {
	Environment* environment = static_cast<Environment*>(context_);
	if( environment == nullptr ) {
		return -1;
	}

	environment->init();

	if( !environment->initRenderer(240, 240, true) ) {
		return -1;
	}

	return 0;
}

void rodent_release(void* context_) {
	Environment* environment = static_cast<Environment*>(context_);
	if( environment == nullptr ) {
		return;
	}

	environment->release();

	delete environment;
}

static void debugSaveFrameImage(const Environment* environment) {
	const void* buffer = environment->getFrameBuffer();
	const char* buf = (const char*)buffer;
	// Write image Y-flipped because OpenGL
	int width = environment->getFrameBufferWidth();
	int height = environment->getFrameBufferHeight();
	stbi_write_png("../debug.png",
				   width, height, 4,
				   buf + (width * 4 * (height - 1)),
				   -width * 4);
}

int rodent_step(void* context_, float* joint_angles) {
	Environment* environment = static_cast<Environment*>(context_);
	if( environment == nullptr ) {
		return -1;
	}

	//..
	for(int i=0; i<8; ++i) {
		printf(">> angle=%f\n", joint_angles[i]);
	}
	//..

	environment->step();

	//..
	debugSaveFrameImage(environment);
	//..
	
	return 0;
}

int rodent_joint_size(void* context_, int* joint_size) {
	Environment* environment = static_cast<Environment*>(context_);
	if( environment == nullptr ) {
		return -1;
	}

	*joint_size = 8; // TODO:
	return 0;
}
