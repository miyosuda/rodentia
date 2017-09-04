#include "BoundingBox.h"
#include <float.h>

#include "Vector3f.h"


/**
 * <!--  BoundingBox():  -->
 */
BoundingBox::BoundingBox()
	:
	minPos(FLT_MAX, FLT_MAX, FLT_MAX),
	maxPos(FLT_MIN, FLT_MIN, FLT_MIN)
{
}

/**
 * <!--  meregeVertex():  -->
 */
void BoundingBox::meregeVertex(float x, float y, float z) {
	if( x < minPos.x ) { minPos.x = x; }
	if( y < minPos.y ) { minPos.y = y; }
	if( z < minPos.z ) { minPos.z = z; }

	if( x > maxPos.x ) { maxPos.x = x; }
	if( y > maxPos.y ) { maxPos.y = y; }
	if( z > maxPos.z ) { maxPos.z = z; }
}

/**
 * <!--  merge():  -->
 */
void BoundingBox::merge(const BoundingBox& boundingBox) {
	if( boundingBox.minPos.x < minPos.x ) { minPos.x = boundingBox.minPos.x; }
	if( boundingBox.minPos.y < minPos.y ) { minPos.y = boundingBox.minPos.y; }
	if( boundingBox.minPos.z < minPos.z ) { minPos.z = boundingBox.minPos.z; }

	if( boundingBox.maxPos.x > maxPos.x ) { maxPos.x = boundingBox.maxPos.x; }
	if( boundingBox.maxPos.y > maxPos.y ) { maxPos.y = boundingBox.maxPos.y; }
	if( boundingBox.maxPos.z > maxPos.z ) { maxPos.z = boundingBox.maxPos.z; }
}

/**
 * <!--  getHalfExtent():  -->
 */
void BoundingBox::getHalfExtent(Vector3f& halfExtent) const {
	halfExtent.sub(maxPos, minPos);
	halfExtent *= 0.5f;
}

/**
 * <!--  getCenter():  -->
 */
void BoundingBox::getCenter(Vector3f& center) const {
	center.add(maxPos, minPos);
	center *= 0.5f;
}

/**
 * <!--  isInitalized():  -->
 */
bool BoundingBox::isInitalized() const {
	return minPos.x > FLT_MAX - 0.00000001f;
}
