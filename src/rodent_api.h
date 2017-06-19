// -*- C++ -*-
#ifndef RODENT_API_HEADER
#define RODENT_API_HEADER

#ifdef __cplusplus
extern "C" {
#endif

void* rodent_create();
int rodent_init(void* context_);
void rodent_release(void* context_);
int rodent_step(void* context_, float* joint_angles);
int rodent_joint_size(void* context_, int* joint_size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
