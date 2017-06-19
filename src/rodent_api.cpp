#include "rodent_api.h"
#include <stdio.h>

#include "Environment.h"

void* rodent_create() {
	Environment* environment = new Environment();
	return static_cast<void*>(environment);
}

int rodent_init(void* context_) {
	Environment* environment = static_cast<Environment*>(context_);
	if( environment == nullptr ) {
		return -1;
	}
	
	return 0;
}

void rodent_release(void* context_) {
	Environment* environment = static_cast<Environment*>(context_);
	if( environment == nullptr ) {
		return;
	}
	
	delete environment;
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
