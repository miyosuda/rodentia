// -*- C++ -*-
#ifndef BOUNDINGBOX_HEADER
#define BOUNDINGBOX_HEADER

#include "Vector3f.h"

class BoundingBox {
private:
	Vector3f minPos;
	Vector3f maxPos;

public:
	BoundingBox();

	void meregeVertex(float x, float y, float z);
	void merge(const BoundingBox& boundingBox);
	void getHalfExtent(Vector3f& halfExtent) const;
	void getCenter(Vector3f& center) const;

	bool isInitalized() const;
};

#endif
