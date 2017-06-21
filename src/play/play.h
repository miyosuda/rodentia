// -*- C++ -*-
#ifndef PLAY_HEADER
#define PLAY_HEADER

enum {
	MOUSE_LEFT_BUTTON,
	MOUSE_RIGHT_BUTTON
};

//void playReshape(int width, int height);
void playMouseDown(int x, int y, int button);
void playMouseDrag(int x, int y, int button);

void playInit(int width, int height);
void playStep();
void playRelease();

#endif
