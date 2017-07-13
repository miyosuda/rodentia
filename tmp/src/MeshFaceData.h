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
	MeshFaceData( const float* vertices_,
				  int verticesSize_,
				  const short* indices_,
				  int indicesSize_ );
	~MeshFaceData();
	
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
