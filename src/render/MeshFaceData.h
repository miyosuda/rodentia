// -*- C++ -*-
#ifndef MESHFACEDATA_HEADER
#define MESHFACEDATA_HEADER

class MeshFaceData {
private:
	float* vertices;
	int verticesSize;
	unsigned short* indices;
	int indicesSize;

public:
	MeshFaceData( const float* vertices_,
				  int verticesSize_,
				  const unsigned short* indices_,
				  int indicesSize_ );
	~MeshFaceData();
	
	const float* getVertices() const {
		return vertices;
	}
	int getVerticesSize() const {
		return verticesSize;
	}
	const unsigned short* getIndices() const {
		return indices;
	}
	int getIndicesSize() const {
		return indicesSize;
	}
};

#endif
