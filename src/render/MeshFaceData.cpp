#include "MeshFaceData.h"

/**
 * <!--  MeshFaceData():  -->
 */
MeshFaceData::MeshFaceData( const float* vertices_,
							int verticesSize_,
							const short* indices_,
							int indicesSize_ )
	:
	verticesSize(verticesSize_),
	indicesSize(indicesSize_) {

	vertices = new float[verticesSize];
	memcpy(vertices, vertices_, sizeof(float) * verticesSize);

	indices = new short[indicesSize];
	memcpy(indices, indices_, sizeof(short) * indicesSize);
}
	
/**
 * <!--  ~MeshFaceData():  -->
 */
MeshFaceData::~MeshFaceData() {
	delete [] vertices;
	delete [] indices;
}
