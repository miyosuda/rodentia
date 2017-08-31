#include "MeshFaceData.h"

#include <gtest/gtest.h>

namespace {
	class MeshFaceDataTest : public ::testing::Test {
	};

	TEST_F( MeshFaceDataTest, allTest) {
		/*
		int verticesSize = 10;
		int indicesSize = 20;
		
		float* vertices = new float[verticesSize];
		unsigned short* indices = new unsigned short[indicesSize];

		for(int i=0; i<verticesSize; ++i) {
			vertices[i] = (float)(100 * i + 1);
		}

		for(int i=0; i<indicesSize; ++i) {
			indices[i] = (unsigned short)(i*2);
		}

		MeshFaceData meshFaceData(vertices, verticesSize, indices, indicesSize);

		// Check whether buffer is copied.
		ASSERT_NE(meshFaceData.getVertices(), vertices);
		ASSERT_NE(meshFaceData.getIndices(), indices);

		// Check buffer size
		ASSERT_EQ(verticesSize, meshFaceData.getVerticesSize());
		ASSERT_EQ(indicesSize, meshFaceData.getIndicesSize());

		delete [] vertices;
		delete [] indices;
		*/
	}
}
