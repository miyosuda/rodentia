#include "LSPSM.h"
#include <float.h>
#include <math.h>

#include "BoundingBox.h"

#define NEW_FORMULA 1

static void createLookTo(const Vector3f& fromPos, const Vector3f& dir,
						 const Vector3f& up,
						 Matrix4f& viewMat) {
	/*
	Vector3f forward(dir);
	forward.normalize();

	Vector3f side;
	side.cross(forward, up);

	if( side.lengthSquared() < 0.00001f ) {
		side.set(1.0f, 0.0f, 0.0f);
	} else {
		side.normalize();
	}

	Vector3f newUp;
	newUp.cross(side, forward);

	Matrix4f mat;
	mat.setZero();
	
	mat.m00 = side.x;
	mat.m10 = side.y;
	mat.m20 = side.z;

	mat.m01 = newUp.x;
	mat.m11 = newUp.y;
	mat.m21 = newUp.z;

	mat.m02 = -forward.x;
	mat.m12 = -forward.y;
	mat.m22 = -forward.z;

	mat.m03 = fromPos.x;
	mat.m13 = fromPos.y;
	mat.m23 = fromPos.z;
	
	mat.m33 = 1.0f;
	viewMat.invertRT(mat);
	*/

	Vector3f zaxis;
	zaxis.set(dir);
	zaxis *= -1.0f; // 反転している
	zaxis.normalize();

	Vector3f xaxis;
	xaxis.cross(up, zaxis);
	xaxis.normalize();

	Vector3f yaxis;
	yaxis.cross(zaxis, xaxis);
	yaxis.normalize();

	// 縦カラム
	viewMat.m00 = xaxis.x;
	viewMat.m10 = yaxis.x;
	viewMat.m20 = zaxis.x;
	viewMat.m30 = 0.0f;

	viewMat.m01 = xaxis.y;
	viewMat.m11 = yaxis.y;
	viewMat.m21 = zaxis.y;
	viewMat.m31 = 0.0f;
	
	viewMat.m02 = xaxis.z;
	viewMat.m12 = yaxis.z;
	viewMat.m22 = zaxis.z;
	viewMat.m32 = 0.0f;

	viewMat.m03 = -xaxis.dot(fromPos);
	viewMat.m13 = -yaxis.dot(fromPos);
	viewMat.m23 = -zaxis.dot(fromPos);
	viewMat.m33 = 1.0f;
}

static float getCrossingAngle(const Vector3f& v0,
							  const Vector3f& v1) {
	float d = v0.length() * v1.length();
	if( d == 0.0f ) {
		return 0.0f;
	}

	float c = v0.dot(v1) / d;
	if( c >= 1.0f ) {
		return 0.0;
	}
	if( c <= -1.0f ) {
		return M_PI;
	}
	return acosf(c);
}

static void transformCoord(const Vector3f& p, const Matrix4f& mat,
						   Vector3f& pout) {
	Vector4f tp;
	mat.transform(Vector4f(p.x, p.y, p.z, 1.0f), tp);
	pout.set(tp.x/tp.w, tp.y/tp.w, tp.z/tp.w);
}


//==============================
//     [VolumePoints Class]
//==============================

VolumePoints::VolumePoints() {
	init();
}

void VolumePoints::init() {
	points[0].set( -1.0f, +1.0f, -1.0f );
	points[1].set( -1.0f, -1.0f, -1.0f );
	points[2].set( +1.0f, -1.0f, -1.0f );
	points[3].set( +1.0f, +1.0f, -1.0f );
	points[4].set( -1.0f, +1.0f, +1.0f );
	points[5].set( -1.0f, -1.0f, +1.0f );
	points[6].set( +1.0f, -1.0f, +1.0f );
	points[7].set( +1.0f, +1.0f, +1.0f );
}

void VolumePoints::transform(const Matrix4f& matrix) {
	for(int i=0; i<POINT_SIZE; ++i) {
		Vector3f& point = points[i];
		transformCoord(point, matrix, point);
	}
}

void VolumePoints::computeBoundingBox(BoundingBox& boundingBox) const {
	for( int i=0; i<POINT_SIZE; ++i ) {
		const Vector3f& p = points[i];
		boundingBox.mergeVertex(p.x, p.y, p.z);
	}
}

VolumePoints& VolumePoints::operator=(const VolumePoints &value) {
	for( int i=0; i<POINT_SIZE; ++i ) {
		points[i].set(value.points[i]);
	}
	return (*this);
}

void VolumePoints::debugDump() const {
	for( int i=0; i<POINT_SIZE; ++i ) {
		points[i].debugDump();
	}
}


//==============================
//           [LSPSM]
//==============================

/**
 * <!--  LSPSM():  -->
 */
LSPSM::LSPSM() {
	nearClip = 0.1f;
}

/**
 * <!--  computeUpVector():  -->
 *
 * 視点・ライトベクトルからアップベクトルを計算する
 *
 * (視点ベクトルとライトベクトルが作る平面上で、ライトベクトルと直交するベクトル)
 */
void LSPSM::computeUpVector(const Vector3f& viewDir,
							const Vector3f& lightDir,
							Vector3f& up) {
	Vector3f left;
	left.cross(lightDir, viewDir);
	up.cross(left, lightDir);
	up.normalize();
}

/**
 * <!--  computeMatrix_USM():  -->
 * 
 * 通常のシャドウマップ行列を計算する
 */
void LSPSM::computeMatrix_USM(VolumePoints& points) {
	// ライトのビュー行列を求める (TODO: この場合、lightDirとviewDirが平行になっているのでは？)
	createLookTo( eyePosition, lightDir, viewDir, lightViewMat );

	// ライト空間へ変換 (ワールド座標系の視錐台の頂点がライト空間へ変換される)
	points.transform(lightViewMat);

	// AABBを算出
	BoundingBox boundingBox;
	points.computeBoundingBox(boundingBox);

	// 範囲を適正にする
	getUnitCubeClipMatrix(boundingBox, lightProjMat);
}

/**
 * <!--  computeMatrix_LSPSM():  -->
 *
 * ライト空間透視シャドウマップ行列を計算
 */
void LSPSM::computeMatrix_LSPSM(float angle, VolumePoints& points) {
	// リストをコピーしておく
	VolumePoints pointsClone = points;

	//float sinGamma = sqrtf(1.0f - angle * angle); // 元のコード
	float sinGamma = sinf(fabsf(angle));

	// アップベクトルを算出 (ライトベクトルと視点ベクトルの平面上で、ライトベクトルに直交する方向)
	Vector3f up;
	computeUpVector(viewDir, lightDir, up);

	// ライトのビュー行列を求める
	// (視点を起点として、ライト方向を向く行列、の逆)
	createLookTo(eyePosition, lightDir, up, lightViewMat);

	// ライトのビュー行列でリストを変換し、AABBを算出
	points.transform(lightViewMat);

	BoundingBox boundingBox;
	points.computeBoundingBox(boundingBox);
	
	Vector3f halfExtent;
	boundingBox.getHalfExtent(halfExtent);

	// 新しい視錐台を求める
	const float factor = 1.0f / sinGamma;
	const float z_n = factor * nearClip;
	const float d = fabsf(halfExtent.y * 2.0f);
	
#if NEW_FORMULA
	//  New Formula written in ShaderX4
	const float z0 = - z_n;
	const float z1 = - ( z_n + d * sinGamma );
	const float n = d / ( sqrtf( z1 / z0 ) - 1.0f );
#else
 	//  Old Formula written in papers
	const float z_f = z_n + d * sinGamma;
	const float n = ( z_n + sqrtf( z_f * z_n ) ) * factor;
#endif

	const float f = n + d;

	// pos = eyePosition - up * (n - nearClip)
	Vector3f upScaled;
	upScaled.scale(n - nearClip, up);
	
	Vector3f pos;
	pos.sub(eyePosition, upScaled);

	// シャドウマップ生成用ライトのビュー行列
	// posを始点として、ライト方向を向く (eyePosから up * (n - nearClip)分ずれている)
	createLookTo(pos, lightDir, up, lightViewMat);

	// Y方向への射影行列を取得
	Matrix4f yProjMat;
	getPerspective(n, f, yProjMat);

	// 透視変換後の空間へ変換する
	Matrix4f tmpLightViewProjMat;
	tmpLightViewProjMat.mul(yProjMat, lightViewMat);
	pointsClone.transform(tmpLightViewProjMat);

	// AABBを算出
	BoundingBox boundingBox2;
	pointsClone.computeBoundingBox(boundingBox2);

	// 範囲を適正にする
	Matrix4f clipMat;
	getUnitCubeClipMatrix(boundingBox2, clipMat);

	// シャドウマップ生成用ライトの射影行列を求める
	lightProjMat.mul(clipMat, yProjMat);
}

/**
 * <!--  updateShadowMatrix():  -->
 *
 * ライトの行列を更新する
 */
void LSPSM::updateShadowMatrix() {
	Matrix4f viewProjMat;
	viewProjMat.mul(eyeProjMat, eyeViewMat);

	// ビューボリュームを求める
	VolumePoints points;
	computeLightVolumePoints(viewProjMat, points);
	
	// ここで視錐台の各頂点にpointsListがなっている
	
	// 視線ベクトルとライトベクトルのなす角度を求める
	float angle = getCrossingAngle(viewDir, lightDir);
	
	// なす角が0度または180度の場合
	if(angle == 0.0f || angle == M_PI) {
		// ライトによる歪みがないので通常のシャドウマップを適用
		computeMatrix_USM(points);
 	} else {
		// Light Space Perspective Shadow Map
		computeMatrix_LSPSM(angle, points);
	}

	// 左手座標系に変換
	Matrix4f scaleMat;
	scaleMat.setIdentity();
	scaleMat.m22 = -1.0f;
	lightProjMat.mul(scaleMat, lightProjMat);
	
	// シャドウマップに使う最終的な行列を求める
	lightViewProjMat.mul(lightProjMat, lightViewMat);
}

/**
 * <!--  computeLightVolumePoints():  -->
 *
 * 凸体を求める
 */
void LSPSM::computeLightVolumePoints(const Matrix4f& viewProjMat, VolumePoints& points) {
	// カメラの視錐台を求める
	computeViewFrustum(viewProjMat, points);

	// カメラの視錐台に影を投げるオブジェクトがあるか判定する
	//  TODO 1: 視錐台と交差しているか？
	/*** 視錐台を形成する平面とAABBとの交差判定を行う ***/

	//  TODO: 2: オブジェクトの各頂点からライトベクトル方向に
	//  レイを飛ばし交差するか？
	/*** 各頂点からライトベクトル方向にレイを飛ばし、視錐台を形成する平面との交差判定を行う ***/
}

/**
 * <!--  computeViewFrustum():  -->
 *
 * カメラの視錐台を求める
 */
void LSPSM::computeViewFrustum(const Matrix4f& viewProjMat,
							   VolumePoints& points) {
	// ビュー行列→射影行列の逆変換を行う行列を求める
	Matrix4f invViewProjMat;
	invViewProjMat.invert(viewProjMat);

	// 立方体に逆変換する行列をかけ、視錐台を求める
	points.transform(invViewProjMat);
}

/**
 * <!--  getUnitCubeClipMatrix():  -->
 * 範囲を適正にする行列を取得する
 */
void LSPSM::getUnitCubeClipMatrix(const BoundingBox& boundingBox,
								  Matrix4f& mat) const {
	Vector3f halfExtent;
	Vector3f center;
	const Vector3f& min = boundingBox.getMinPos();
	boundingBox.getHalfExtent(halfExtent);
	boundingBox.getCenter(center);
	
	/*
	// 新しいコード
	float sx =  1.0f / halfExtent.x;
	float sy =  1.0f / halfExtent.y;
	float sz = -1.0f / halfExtent.z; // TODO: ここマイナス入っていなかった & x0.5されている
	float tx = -center.x / halfExtent.x;
	float ty = -center.y / halfExtent.y;
	float tz =  min.z / halfExtent.z; // TODO: x0.5されている

	mat.m00 = sx;    mat.m01 = 0.0f;  mat.m02 = 0.0f;  mat.m03 = tx;
	mat.m10 = 0.0f;  mat.m11 = sy;    mat.m12 = 0.0f;  mat.m13 = ty;
	mat.m20 = 0.0f;  mat.m21 = 0.0f;  mat.m22 = sz;    mat.m23 = tz;
	mat.m30 = 0.0f;  mat.m31 = 0.0f;  mat.m32 = 0.0f;  mat.m33 = 1.0f;
	*/
	
	// 以前のコード
	// Row方向
	mat.m00 = 1.0f / halfExtent.x;
	mat.m01 = 0.0f;
	mat.m02 = 0.0f;
	mat.m03 = -center.x / halfExtent.x;

	mat.m10 = 0.0f;
	mat.m11 = 1.0f / halfExtent.y;
	mat.m12 = 0.0f;
	mat.m13 = -center.y / halfExtent.y;

	mat.m20 = 0.0f;
	mat.m21 = 0.0f;
	mat.m22 =   1.0f / ( halfExtent.z * 2.0f );
	mat.m23 = -min.z / ( halfExtent.z * 2.0f );

	mat.m30 = 0.0f;
	mat.m31 = 0.0f;
	mat.m32 = 0.0f;
	mat.m33 = 1.0f;
}

/**
 * <!--  getPerspective():  -->
 *
 * Y方向への射影行列を取得する
 */
void LSPSM::getPerspective(const float near, const float far, Matrix4f& mat) const {
	mat.setIdentity();
	mat.m11 = far / (far - near);
	mat.m31 = 1.0f;
	mat.m13 = -far * near / (far - near);
	mat.m33 = 0.0f;
}
