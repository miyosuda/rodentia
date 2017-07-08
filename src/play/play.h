// -*- C++ -*-
#ifndef PLAY_HEADER
#define PLAY_HEADER

enum {
	MOUSE_LEFT_BUTTON = 0,
	MOUSE_RIGHT_BUTTON
};

enum {
	KEY_ACTION_LOOK_LEFT = 0,
	KEY_ACTION_LOOK_RIGHT,
	KEY_ACTION_STRAFE_LEFT,
	KEY_ACTION_STRAFE_RIGHT,	
	KEY_ACTION_MOVE_FORWARD,
	KEY_ACTION_MOVE_BACKWARD,
};

//void playReshape(int width, int height);
void playMouseDown(int x, int y, int button);
void playMouseDrag(int x, int y, int button);
void playKey(int actionKey, bool press);

void playInit(int width, int height);
void playStep();
void playRelease();

#endif
