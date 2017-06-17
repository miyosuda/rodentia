#include "play.h"
#include <assert.h>
#include "Graphics.h"
#include "TrackBall.h"

TrackBall trackBall(0.0f, 0.0f, -800.0f, -0.3f);

// platform specificÇ»ä÷êî===========================================
/**
 * reshape():
 */
void playReshape(int width, int height) {
	trackBall.resize(width, height);
}

/**
 * mouseDown():
 */
void playMouseDown(int x, int y, int button) {
	if(button == MOUSE_LELFT_BUTTON) {
		trackBall.startRotation(x, y);
	} else if(button == MOUSE_RIGHT_BUTTON) {
		trackBall.startZoom(x, y);
	}
	//trackBall.startTrans(x, y);
}

/**
 * mouseDrag():
 */
void playMouseDrag(int x, int y, int button) {
	if(button == MOUSE_LELFT_BUTTON) {
		trackBall.dragRotation(x, y);
	} else if(button == MOUSE_RIGHT_BUTTON) {
		trackBall.dragZoom(x, y);
	}
	//trackBall.dragTrans(x, y);
}

//===============================================================
/**
 * playInit():
 */
void playInit() {
	Graphics::getGraphics().init();

	//ModelSet::init(filename);
}

/**
 * playLoop():
 */
void playLoop() {
	Matrix4f camera;
	trackBall.getMat(camera);

	Graphics::getGraphics().setCamera( camera );

	/*
	ModelSet::update();
	ModelSet::draw();
	Renderer::renderGrid();
	Status::endFrame();
	
	sprintf(buffer, "generation %d", ModelSet::getGeneration()+1);
	Renderer::drawString( buffer,
						  -400.0f, -148.0f, 300.0f,
						  3.0f );	
	*/
}

/**
 * playFinalize():
 */
void playFinalize() {
//	ModelSet::release();
}
