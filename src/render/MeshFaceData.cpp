#include "MeshFaceData.h"
#include <string.h>

/**
 * <!--  MeshFaceData():  -->
 */
MeshFaceData::MeshFaceData( const float* vertices_,
							int verticesSize_,
							const unsigned short* indices_,
							int indicesSize_ )
	:
	verticesSize(verticesSize_),
	indicesSize(indicesSize_) {

	vertices = new float[verticesSize];
	memcpy(vertices, vertices_, sizeof(float) * verticesSize);

	indices = new unsigned short[indicesSize];
	memcpy(indices, indices_, sizeof(short) * indicesSize);
}
	
/**
 * <!--  ~MeshFaceData():  -->
 */
MeshFaceData::~MeshFaceData() {
	delete [] vertices;
	delete [] indices;
}
