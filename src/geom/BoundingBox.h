// -*- C++ -*-
#ifndef BOUNDINGBOX_HEADER
#define BOUNDINGBOX_HEADER

#include "Vector3f.h"

class Matrix4f;


class BoundingBox {
private:
    Vector3f minPos;
    Vector3f maxPos;

public:
    BoundingBox();

    void set(const Vector3f& minPos_, const Vector3f& maxPos_);
    void mergeVertex(float x, float y, float z);
    void merge(const BoundingBox& boundingBox);
    void getHalfExtent(Vector3f& halfExtent) const;
    void getCenter(Vector3f& center) const;
    
    const Vector3f& getMinPos() const { return minPos; }
    const Vector3f& getMaxPos() const { return maxPos; }
    
    void transform(float scaleX, float scaleY, float scaleZ,
                   const Matrix4f& mat,
                   BoundingBox& outBoundingBox) const;

    bool isInitalized() const;

    void debugDump() const {
        printf("[%f %f %f] - [%f %f %f]]\n",
               minPos.x, minPos.y, minPos.z,
               maxPos.x, maxPos.y, maxPos.z);
    }   
};

#endif
