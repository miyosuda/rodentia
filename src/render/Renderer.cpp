#include "Renderer.h"
#include "glinc.h"

/**
 * <!--  setCameraMat():  -->
 */
void Renderer::setCameraMat(const Matrix4f& mat) {
	camera.setMat(mat);
}

// Upside-Down flip matrix
/*
static const Matrix4f flipMat( 1.0f, 0.0f, 0.0f, 0.0f,
							   0.0f,-1.0f, 0.0f, 0.0f,
							   0.0f, 0.0f, 1.0f, 0.0f,
							   0.0f, 0.0f, 0.0f, 1.0f );
*/

/**
 * <!--  renderPre():  -->
 */
void Renderer::renderPre() {
	// TODO:
	/*
	Matrix4f worldCamera;
	worldCamera.invertRT(camera);

	glMatrixMode(GL_MODELVIEW);

	if( flipping ) {
		// flip upside down
		Matrix4f mat(flipMat);
		mat *= worldCamera;
		glLoadMatrixf(mat.getPointer());
	} else {
		// normal
		glLoadMatrixf(worldCamera.getPointer());
	}
	*/
	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}
