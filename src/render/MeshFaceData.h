// -*- C++ -*-
#ifndef MESHFACEDATA_HEADER
#define MESHFACEDATA_HEADER

#include "Vector3f.h"
#include "BufferObjects.h"

class MeshFaceData {
private:
	int verticesSize;
	int indicesSize;

	Vector3f minPos;
	Vector3f maxPos;

	VertexArray vertexArray;
	VertexBuffer vertexBuffer;
	IndexBuffer indexBuffer;

	VertexArray depthVertexArray;

	void release();

public:
	MeshFaceData();
	bool init(const float* vertices,
			  int verticesSize_,
			  const unsigned short* indices,
			  int indicesSize_ );
	~MeshFaceData();
	void calcBoundingBox(Vector3f& minPos, Vector3f& maxPos) const;
	void draw(bool forShadow) const;
};

#endif
