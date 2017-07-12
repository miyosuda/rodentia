// -*- C++ -*-
#ifndef MESHFACEDATA_HEADER
#define MESHFACEDATA_HEADER

#include <string.h>

class MeshFaceData {
private:
	float* vertices;
	int verticesSize;
	short* indices;
	int indicesSize;

public:
	MeshFaceData( float* vertices_,
				  int verticesSize_,
				  short* indices_,
				  int indicesSize_ )
		:
		verticesSize(verticesSize_),
		indicesSize(indicesSize_) {

		vertices = new float[verticesSize];
		memcpy(vertices, vertices_, sizeof(float)*verticesSize);

		indicesSize = indicesSize_;
		indices = new short[indicesSize];
		memcpy(indices, indices_, sizeof(short)*indicesSize);
	}
	
	~MeshFaceData() {
		delete [] vertices;
		delete [] indices;
	}
	
	const float* getVertices() const {
		return vertices;
	}
	int getVerticesSize() const {
		return verticesSize;
	}
	const short* getIndices() const {
		return indices;
	}
	int getIndicesSize() const {
		return indicesSize;
	}
};

#endif
