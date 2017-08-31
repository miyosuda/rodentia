#include "MeshFaceData.h"
#include <string.h>
#include <float.h>

/**
 * <!--  MeshFaceData():  -->
 */
MeshFaceData::MeshFaceData() {
}

/**
 * <!--  init():  -->
 */
bool MeshFaceData::init( const float* vertices,
						 int verticesSize_,
						 const unsigned short* indices,
						 int indicesSize_ ) {
	verticesSize = verticesSize_;
	indicesSize = indicesSize_;
	
	// Set min max pos for bounding box.
	minPos.set(FLT_MAX, FLT_MAX, FLT_MAX);
	maxPos.set(FLT_MIN, FLT_MIN, FLT_MIN);		
	
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

	// Set gl buffer objects

	// TODO: Add errro check
	
	indexBuffer.init(indices, indicesSize);
	
	vertexArray.init();
	vertexBuffer.init(vertices, verticesSize);
	vertexBuffer.bind();

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4*8, (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 4*8, (void*)(4*3));
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 4*8, (void*)(4*6));
	
	vertexBuffer.unbind();
	vertexArray.unbind();

	return true;
}

/**
 * <!--  release():  -->
 */
void MeshFaceData::release() {
	vertexBuffer.release();
	indexBuffer.release();
	vertexArray.release();
}

/**
 * <!--  draw():  -->
 */
void MeshFaceData::draw() const {
	vertexArray.bind();
	indexBuffer.bind();
	
	glDrawElements( GL_TRIANGLES, indicesSize, GL_UNSIGNED_SHORT, 0 );
	
	vertexArray.unbind();

	indexBuffer.unbind();
}

/**
 * <!--  ~MeshFaceData():  -->
 */
MeshFaceData::~MeshFaceData() {
	release();
}

/**
 * <!--  calcBoundingBox():  -->
 */
void MeshFaceData::calcBoundingBox(Vector3f& minPos_, Vector3f& maxPos_) const {
	minPos_.set(minPos);
	maxPos_.set(maxPos);
}
