#include "Graphics.h"
#include <stdio.h>
#include "Vector4f.h"
#include "Matrix4f.h"
//#include <GL/gl.h>
#include <GLUT/glut.h>

#include <assert.h>

/*
 [axis]
       [z]
       /
      /
     *------[x]
     |
     |
    [y]


 [indices]
     0------1
    /|     /|
   / |    / |
  4--+---5  |       
  |  3---|--2
  | /    | /
  |/     |/
  7------6 


 [edge]
  +------+  +------+
  |5    1|  |1----0|
  || +x ||  |  +z  |
  |6    2|  |2----3|
  +------+  +------+
            +------+  +------+
            |2    3|  |3----0|
            || +y ||  |  -x  |
            |6    7|  |7----4|
            +------+  +------+
                      +------+  +------+
                      |7    4|  |4----0|
                      || -z ||  |  -y  |
                      |6    5|  |5----1|
                      +------+  +------+
*/

static Vector4f vertices[8] = {
	Vector4f(-1.0f, -1.0f, +1.0f, 1.0f),
	Vector4f(+1.0f, -1.0f, +1.0f, 1.0f),
	Vector4f(+1.0f, +1.0f, +1.0f, 1.0f),
	Vector4f(-1.0f, +1.0f, +1.0f, 1.0f),
	Vector4f(-1.0f, -1.0f, -1.0f, 1.0f),
	Vector4f(+1.0f, -1.0f, -1.0f, 1.0f),
	Vector4f(+1.0f, +1.0f, -1.0f, 1.0f),
	Vector4f(-1.0f, +1.0f, -1.0f, 1.0f)
};

static u_int indices[24] = {
	0, 3, 1, 2, // +z
	2, 3, 6, 7, // +y
	2, 6, 1, 5, // +x
	5, 6, 4, 7, // -z
	0, 1, 4, 5, // -y
	4, 7, 0, 3, // -x
};

static Vector4f normals[6] = {
	Vector4f( 0.0f,  0.0f,  1.0f, 0.0f), // +z
	Vector4f( 0.0f,  1.0f,  0.0f, 0.0f), // +y 
	Vector4f( 1.0f,  0.0f,  0.0f, 0.0f), // +x
	Vector4f( 0.0f,  0.0f, -1.0f, 0.0f), // -z
	Vector4f( 0.0f, -1.0f,  0.0f, 0.0f), // -y
	Vector4f(-1.0f,  0.0f,  0.0f, 0.0f), // -x
};

static u_int edges[24] = {
	1, 0, 2, 3, // +z
	2, 6, 3, 7, // +y
	5, 6, 1, 2, // +x
	7, 6, 4, 5, // -z
	4, 0, 5, 1, // -y
	3, 0, 7, 4, // -x
};

Graphics Graphics::g; // シングルトンのインスタンス

/**
 * init():
 */
void Graphics::init() {
	//	sceVu0ViewScreenMatrix(m, scrz, ax, ay, cx, cy, zmin, zmax, nearz, farz)
	// 	m           出力:マトリックス
	//	scrz        入力:(スクリーンまでの距離)
	//	ax          入力:(Ｘ方向アスペクト比)
	//	ay          入力:(Ｙ方向アスペクト比)
	//	cx          入力:(スクリーンの中心Ｘ座標)
	//	cy          入力:(スクリーンの中心Ｙ座標)
	//	zmin        入力:(Ｚバッファ最小値)
	//	zmax        入力:(Ｚバッファ最大値)
	//	nearz       入力:(ニアクリップ面のＺ)
	//	farz        入力:(ファークリップ面のＺ)
/*
	sceVu0ViewScreenMatrix( cameraScreen.getSCEPointer(),
							512.0f, 1.0f, 0.47f, 
							2048.0f, 2048.0f, 1.0f, 
							16777215.0f, 1.0f, 65536.0f );
*/
	// 0xffffffff = 4294967295 (32bit)
	// 0x00ffffff = 16777215   (24bit)
	// 0x00010000 = 65536
	// 0x00000040 = 64

/*
	Vector4f light0(  0.0f,  1.5f, 0.5f, 0.0f );
	Vector4f light1(  1.5f, -0.5f, 0.5f, 0.0f );
	Vector4f light2( -1.5f, -0.5f, 0.5f, 0.0f );
	sceVu0NormalLightMatrix( lightDirs.getSCEPointer(), 
							 light0.getPointer(), 
							 light1.getPointer(), 
							 light2.getPointer() );	
*/

	Vector4f color0(0.3f, 0.3f, 0.3f, 0.0f); // wは0.0fでいいハズ.
	Vector4f color1(0.8f, 0.8f, 0.8f, 0.0f); // 
	Vector4f color2(0.3f, 0.3f, 0.3f, 0.0f); // 
	Vector4f ambient(0.1f, 0.1f, 0.1f, 0.0f);
/*

	sceVu0LightColorMatrix( lightColors.getSCEPointer(), 
							color0.getPointer(),
							color1.getPointer(),
							color2.getPointer(),
							ambient.getPointer());	
*/

	// 単位行列だと原点から+z方向を眺めていることになる.
	camera.setIdentity();

	shadow.setIdentity();

	setShadow( Vector4f(-0.6f, 1.0f, -0.6f, 0.0f) );

	// ここでsetAlphaも呼ぶようにする.
	// ==> それとも明示的に呼ぶようにするかどうか
	//     検討すること.
	setAlpha();
}

/**
 * setAlpha():
 */
void Graphics::setAlpha() {
}

/**
 * setCamera():
 */
void Graphics::setCamera(const Matrix4f& camera_) {
	camera = camera_;
	worldCamera.invertRT(camera);

	shadowCamera.mul(worldCamera, shadow);
}

/**
 * <!--  setShadow():  -->
 *
 * 射影平面はy=0固定でライトの向きだけ指定.
 */
void Graphics::setShadow(const Vector4f& dir) {
	shadow.setIdentity();
	assert(dir.y != 0.0f);
	shadow.setColumn( 1, Vector4f(dir.z/dir.y, 0.0f, dir.x/dir.y, 0.0f) );
	// 若干床下に影を下げる
	shadow.setColumn( 3, Vector4f(0.0, 0.01f, 0.0f, 1.0f) );
	shadowCamera.mul(worldCamera, shadow);
}

/**
 * drawBox():
 */
void Graphics::drawBox( const Matrix4f& mat, const Vector4f& range, 
						const Vector4f& color ) {
	Vector4f vert[8];
//	Vector4f norm[6];

	Vector4f shad[8];

	int i, j;
	for(i=0; i<8; ++i) {
		Vector4f& v = vertices[i];
		vert[i].parallelProduct(v, range);
		mat.transform(vert[i], vert[i]);
		shadowCamera.transform(vert[i], shad[i]);
		worldCamera.transform(vert[i], vert[i]);
	}	

	for(i=0; i<6; ++i) {
		glColor3f(color.x, color.y, color.z);	
		glBegin(GL_TRIANGLE_STRIP);
		// face vertex
		for(j=0; j<4; ++j) {
			Vector4f& v = vert[ indices[4*i + j] ];
			glVertex3f(v.x, v.y, v.z);			
		}
		glEnd();

		// edge lines
		glColor3f(0.0f, 0.0f, 0.0f);
		glBegin(GL_LINES);
		for(j=0; j<4; ++j) {
			Vector4f& v = vert[ edges[4*i + j] ];
			glVertex3f(v.x, v.y, v.z);
		}
		glEnd();		

		// shadow
		//glColor3f(0.0f, 0.0f, 0.0f);
		//glColor3f(0.5f, 0.5f, 0.5f);
		glColor3f(0.7f, 0.7f, 0.7f);
		glBegin(GL_TRIANGLE_STRIP);
		// shadow vertex
		for(j=0; j<4; ++j) {
			Vector4f& v = shad[ indices[4*i + j] ];
			glVertex3f(v.x, v.y, v.z);
		}
		glEnd();
	}
}

/**
 * drawLine():
 */
void Graphics::drawLine( const Vector4f& pos0, 
						 const Vector4f& pos1, 
						 const Vector4f& color ) {

	Vector4f pos0_;
	Vector4f pos1_;

	worldCamera.transform(pos0, pos0_);
	worldCamera.transform(pos1, pos1_);

	glColor3f(color.x, color.y, color.z);	
	glBegin(GL_LINES);
	glVertex3f(pos0_.x, pos0_.y, pos0_.z);
	glVertex3f(pos1_.x, pos1_.y, pos1_.z);
	glEnd();
}

#define ARC_DIVISION 12

/**
 * <!--  drawArc():  -->
 */
void Graphics::drawArc( const Matrix4f& mat,
						float r,
						float angleMin,
						float angleMax,
						const Vector4f& color ) {

	// 以下頂点データ作成
	const Vector4f& center = mat.getColumnRef(3);
	const Vector4f& axisZ  = mat.getColumnRef(2);
	const Vector4f& axisX  = mat.getColumnRef(0);

	float deltaAngle = (angleMax - angleMin) / (float)(ARC_DIVISION);

	// 頂点は、ARC_DIVISION + 3個になる.
	Vector4f pos;
	Vector4f v;
	int i;

	glColor3f(color.x, color.y, color.z);

	glBegin(GL_LINE_STRIP);
	worldCamera.transform(center, v);	
	glVertex3f(v.x, v.y, v.z);

	for(i=0; i<=ARC_DIVISION; ++i) {	
		float angle = angleMin + i * deltaAngle;
		pos.scaleAdd( r * cosf(angle), 
					  axisZ, center );
		pos.scaleAdd( r * sinf(angle), 
					  axisX );

		worldCamera.transform(pos, v);
		glVertex3f(v.x, v.y, v.z);
	}	

	worldCamera.transform(center, v);
	glVertex3f(v.x, v.y, v.z);

	glEnd();
}

