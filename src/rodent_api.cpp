#include "rodent_api.h"
#include <stdio.h>

class Context {
};

void* rodent_create() {
	Context* context = new Context();
	return static_cast<void*>(context);
}

int rodent_init(void* context_) {
	Context* context = static_cast<Context*>(context_);
	if( context == nullptr ) {
		return -1;
	}
	
	return 0;
}

void rodent_release(void* context_) {
	Context* context = static_cast<Context*>(context_);
	if( context == nullptr ) {
		return;
	}
	
	delete context;
}

int rodent_step(void* context_, float* joint_angles) {
	Context* context = static_cast<Context*>(context_);
	if( context == nullptr ) {
		return -1;
	}

	//..
	for(int i=0; i<8; ++i) {
		printf(">> angle=%f\n", joint_angles[i]);
	}
	//..
	
	return 0;
}

int rodent_joint_size(void* context_, int* joint_size) {
	Context* context = static_cast<Context*>(context_);
	if( context == nullptr ) {
		return -1;
	}

	*joint_size = 8; // TODO:
	return 0;
}
