// -*- C++ -*-
#ifndef PLAY_HEADER
#define PLAY_HEADER

#include "common.h"

enum {
	MOUSE_LELFT_BUTTON,
	MOUSE_RIGHT_BUTTON
};

// windows specificÇ»ä÷êî
void playReshape(int width, int height);
void playMouseDown(int x, int y, int button);
void playMouseDrag(int x, int y, int button);

void playInit();
void playLoop();
void playFinalize();

#endif
