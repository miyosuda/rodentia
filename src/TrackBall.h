#ifndef TRACKBALL_HEADER
#define TRACKBALL_HEADER

#include "common.h"
#include "Vector4f.h"
#include "Matrix4f.h"

//=====================
//     [TrackBall]
//=====================
class TrackBall {
private:
	float eyex, eyey, eyez;
	float angle;
	Quat4f   q;
	Matrix4f trans;

	Vector4f lastSpherialPos; // ‹…–Ê‚Émap‚³‚ê‚½“_(w=0)
	int lastX, lastY;
	int width, height;
	void sphericalMap(int x, int y, Vector4f& sphericalPos);
	void calcAxisAngle(int x, int y, 
					   Vector4f& axis,
					   float& angle);

public:
	TrackBall(float eyex_, float eyey_, float eyez_,
			  float pitchAngle = 0.0f);
	void resize(int width_, int height_);

	void startRotation(int x, int y);
	void startZoom(int x, int y);
	void startTrans(int x, int y);
	void dragRotation(int x, int y);
	void dragZoom(int x, int y);
	void dragTrans(int x, int y);
	void getMat(Matrix4f& mat);

	void dump();
};

#endif




