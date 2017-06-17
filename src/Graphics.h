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
	Graphics() {}

	static Graphics g; // singleton

public:
	void init();
	void setCamera(const Matrix4f& camera_);
	void drawLine( const Vector4f& pos0, 
				   const Vector4f& pos1, 
				   const Vector4f& color );
	
	static Graphics& getGraphics() { return g; }
};

#endif
