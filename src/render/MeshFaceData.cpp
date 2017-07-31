#include "MeshFaceData.h"
#include <string.h>

#include "Vector3f.h"

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

/**
 * <!--  calcBoundingBox():  -->
 */
void MeshFaceData::calcBoundingBox(Vector3f& minPos, Vector3f& maxPos) const {

	for(int i=0; i<verticesSize/8; ++i) {
		float vx = vertices[8*i+0];
		float vy = vertices[8*i+1];
		float vz = vertices[8*i+2];
	
		if( vx < minPos.x ) { minPos.x = vx; }
		if( vy < minPos.y ) { minPos.y = vy; }
		if( vz < minPos.z ) { minPos.z = vz; }
	
		if( vx > maxPos.x ) { maxPos.x = vx; }
		if( vy > maxPos.y ) { maxPos.y = vy; }
		if( vz > maxPos.z ) { maxPos.z = vz; }
	}
}
