// -*- C++ -*-
#ifndef MESHFACEDATA_HEADER
#define MESHFACEDATA_HEADER

#include "glinc.h"

class VertexArray {
private:
	GLuint handle;

public:
	VertexArray()
		:
		handle(0) {
	}
	
	~VertexArray() {
		release();
	}

	void release() {
		if( handle > 0 ) {
			glDeleteVertexArrays(1, &handle);
			handle = 0;
		}
	}
	
	void init() {
		glGenVertexArrays(1, &handle);
		bind();
	}

	void bind() const {
		glBindVertexArray(handle);
	}

	void unbind() const {
		glBindVertexArray(0);
	}
};


class VertexBuffer {
private:
	GLuint handle;

public:
	VertexBuffer()
		:
		handle(0) {
	}
	
	~VertexBuffer() {
		release();
	}

	void release() {
		if( handle > 0 ) {
			glDeleteBuffers(1, &handle);
			handle = 0;
		}
	}
	
	void init(const float* array, int arraySize) {
		glGenBuffers(1, &handle);
		bind();
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * arraySize,
					 array, GL_STATIC_DRAW);
	}

	void bind() const {
		glBindBuffer(GL_ARRAY_BUFFER, handle);
	}

	void unbind() const {
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
};

class IndexBuffer {
private:
	GLuint handle;

public:
	IndexBuffer()
		:
		handle(0) {
	}
	
	~IndexBuffer() {
		release();
	}

	void release() {
		if( handle > 0 ) {
			glDeleteBuffers(1, &handle);
			handle = 0;
		}
	}
	
	void init(const unsigned short* array, int arraySize) {
		glGenBuffers(1, &handle);
		bind();
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned short) * arraySize,
					 array, GL_STATIC_DRAW);
	}

	void bind() const{
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, handle);
	}

	void unbind() const {
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}
};

#include "Vector3f.h"

class MeshFaceData {
private:
	int verticesSize;
	int indicesSize;

	Vector3f minPos;
	Vector3f maxPos;

	VertexArray vertexArray;
	VertexBuffer vertexBuffer;
	IndexBuffer indexBuffer;

	void release();

public:
	MeshFaceData();
	bool init(const float* vertices,
			  int verticesSize_,
			  const unsigned short* indices,
			  int indicesSize_ );
	~MeshFaceData();
	void calcBoundingBox(Vector3f& minPos, Vector3f& maxPos) const;
	void draw() const;
	
	/*
	int getVerticesSize() const {
		return verticesSize;
	}
	int getIndicesSize() const {
		return indicesSize;
	}
	*/
};

#endif
