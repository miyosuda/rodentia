#include "BoundingBox.h"
#include <float.h>

#include "Matrix4f.h"


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
 * <!--  set():  -->
 */
void BoundingBox::set(const Vector3f& minPos_, const Vector3f& maxPos_) {
    minPos.set(minPos_);
    maxPos.set(maxPos_);
}

/**
 * <!--  mergeVertex():  -->
 */
void BoundingBox::mergeVertex(float x, float y, float z) {
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
 * <!--  transform():  -->
 */
void BoundingBox::transform(float scaleX, float scaleY, float scaleZ,
                            const Matrix4f& mat,
                            BoundingBox& outBoundingBox) const {
    Vector3f halfExtent;
    Vector3f center;

    getHalfExtent(halfExtent);
    getCenter(center);

    halfExtent.x *= scaleX;
    halfExtent.y *= scaleY;
    halfExtent.z *= scaleZ;

    Vector3f scaledMinPos;
    Vector3f scaledMaxPos;  
    scaledMinPos.sub(center, halfExtent);
    scaledMaxPos.add(center, halfExtent);

    Vector4f trans;
    mat.getColumn(3, trans);
    
    Vector3f newMinPos(trans.x, trans.y, trans.z);
    Vector3f newMaxPos(trans.x, trans.y, trans.z);

    float* newMinPos_ = newMinPos.getPointer();
    float* newMaxPos_ = newMaxPos.getPointer();

    const float* scaledMinPos_ = scaledMinPos.getPointer();
    const float* scaledMaxPos_ = scaledMaxPos.getPointer();

    for(int j=0; j<3; ++j ) {
        for(int i=0; i<3; ++i ) {
            float a = mat.getElement(i,j) * scaledMinPos_[i];
            float b = mat.getElement(i,j) * scaledMaxPos_[i];

            if( a < b ) {
                newMinPos_[j] += a;
                newMaxPos_[j] += b;
            } else {
                newMinPos_[j] += b;
                newMaxPos_[j] += a;
            }
        }
    }

    outBoundingBox.set(newMinPos, newMaxPos);
}

/**
 * <!--  isInitalized():  -->
 */
bool BoundingBox::isInitalized() const {
    return minPos.x < FLT_MAX - 0.00000001f;
}
