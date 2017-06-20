#include "rodent_api.h"
#include <stdio.h>

#include "Environment.h"

//..
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
//..

//..
#include "OffscreenRenderer.h"
static OffscreenRenderer renderer;
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

	// MEMO: width, heightはframeバッファサイズの設定と、projectionのaspect計算に使われている.
	renderer.init(240, 240);
	
	return 0;
}

void rodent_release(void* context_) {
	Environment* environment = static_cast<Environment*>(context_);
	if( environment == nullptr ) {
		return;
	}

	environment->release();

	renderer.release();
	
	delete environment;
}

static void debugSaveFrameImage() {
	const void* buffer = renderer.getBuffer();
	const char* buf = (const char*)buffer;
	// Write image Y-flipped because OpenGL
	int width = renderer.getFrameBufferWidth();
	int height = renderer.getFrameBufferHeight();
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

	renderer.renderPre();

	environment->step();

	renderer.render();

	//..
	debugSaveFrameImage();
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
