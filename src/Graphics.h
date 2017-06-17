// -*- C++ -*-
#ifndef GRAPHICS_HEADER
#define GRAPHICS_HEADER

#include "common.h"
#include "Matrix4f.h"

//=====================
//     [Graphics]
//=====================
class Graphics {
private:
	Matrix4f camera;
	Matrix4f worldCamera;
	Matrix4f lightDirs;
	Matrix4f lightColors;
	Matrix4f shadow;
	Matrix4f shadowCamera;
	Graphics() {}

	static Graphics g; // singleton

public:
	void init();
	void setAlpha();
	void setCamera(const Matrix4f& camera_);
	void setShadow(const Vector4f& dir);
	void drawBox( const Matrix4f& mat, const Vector4f& range, 
				  const Vector4f& color );
	void drawLine( const Vector4f& pos0, 
				   const Vector4f& pos1, 
				   const Vector4f& color );
	void drawArc( const Matrix4f& mat,
				  float r,
				  float angleMin,
				  float angleMax,
				  const Vector4f& color );

	static Graphics& getGraphics() { return g; }
};

#endif
