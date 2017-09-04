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

	void transform(float scaleX, float scaleY, float scaleZ,
				   const Matrix4f& mat,
				   BoundingBox& outBoundingBox) const;

	bool isInitalized() const;
};

#endif
