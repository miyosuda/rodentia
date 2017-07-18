#include "MeshManager.h"
#include "Mesh.h"
#include "MeshFace.h"
#include "MeshFaceData.h"

#include <gtest/gtest.h>

namespace {
	class MeshManagerTest : public ::testing::Test {
	};


	TEST_F( MeshManagerTest, getBoxMesh) {
		MeshManager meshManager;
		const Mesh* mesh0 = meshManager.getBoxMesh(nullptr);

		// Check mesh size
		int meshFaceSize = mesh0->debugGetMeshFaceSize();
		ASSERT_EQ(1, meshFaceSize);
		
		const MeshFace* meshFace0 = mesh0->debugGetMeshFace(0);
		const MeshFaceData& meshFaceData0 =
			meshFace0->debugGetMeshFaceData();

		int verticesSize = meshFaceData0.getVerticesSize();
		int indicesSize = meshFaceData0.getIndicesSize();

		// Check vertex and index size
		ASSERT_EQ(verticesSize, 4*6*8);
		ASSERT_EQ(indicesSize, 6*6);

		// Check vetex range
		const float* vertices = meshFaceData0.getVertices();
		for(int i=0; i<verticesSize; ++i) {
			float f = vertices[i];
			ASSERT_LE(f, 1.0f);
			ASSERT_GE(f, -1.0f);
		}

		// Check index range
		const short* indices = meshFaceData0.getIndices();
		
		for(int i=0; i<indicesSize; ++i) {
			short index = indices[i];
			ASSERT_LT(index, 6*6);
			ASSERT_GE(index, 0);
		}
		
		// Same MeshFaceData instance with cache.
		const Mesh* mesh1 = meshManager.getBoxMesh(nullptr);
		const MeshFace* meshFace1 = mesh1->debugGetMeshFace(0);
		
		const MeshFaceData& meshFaceData1 =
			meshFace1->debugGetMeshFaceData();
		
		ASSERT_EQ(&meshFaceData0, &meshFaceData1);
	}
	
	TEST_F( MeshManagerTest, getSphereMesh) {
		MeshManager meshManager;
		const Mesh* mesh0 = meshManager.getSphereMesh(nullptr);

		// Check mesh size
		int meshFaceSize = mesh0->debugGetMeshFaceSize();
		ASSERT_EQ(1, meshFaceSize);
		
		const MeshFace* meshFace0 = mesh0->debugGetMeshFace(0);
		const MeshFaceData& meshFaceData0 =
			meshFace0->debugGetMeshFaceData();

		int verticesSize = meshFaceData0.getVerticesSize();
		int indicesSize = meshFaceData0.getIndicesSize();

		// Check vertex and index size
		ASSERT_EQ(verticesSize, 20*20*8);
		ASSERT_EQ(indicesSize, (20-1)*(20-1)*6);

		// Check vetex range
		const float* vertices = meshFaceData0.getVertices();
		for(int i=0; i<verticesSize; ++i) {
			float f = vertices[i];
			ASSERT_LE(f, 1.0f);
			ASSERT_GE(f, -1.0f);
		}

		// Check index range
		const short* indices = meshFaceData0.getIndices();
		
		for(int i=0; i<indicesSize; ++i) {
			short index = indices[i];
			ASSERT_LT(index, 20*20);
			ASSERT_GE(index, 0);
		}
		
		// Same MeshFaceData instance with cache.
		const Mesh* mesh1 = meshManager.getSphereMesh(nullptr);
		const MeshFace* meshFace1 = mesh1->debugGetMeshFace(0);
		
		const MeshFaceData& meshFaceData1 =
			meshFace1->debugGetMeshFaceData();
		
		ASSERT_EQ(&meshFaceData0, &meshFaceData1);
	}
}
