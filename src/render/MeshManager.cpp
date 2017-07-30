#include "MeshManager.h"

#include <math.h>

#include "MeshFace.h"
#include "MeshData.h"
#include "MeshFaceData.h"
#include "Mesh.h"
#include "Material.h"

/*
    [y]
     |
     |
     |
     *------[x]
    /
   /
 [z]
*/

static float boxVertices[] = {	
	// +z
	-1.0f, -1.0f,  1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, // left bottom
	 1.0f, -1.0f,  1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, // right bottom
	 1.0f,  1.0f,  1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, // right top
	-1.0f,  1.0f,  1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // left top
  
	// -z
	-1.0f, -1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f,
	-1.0f,  1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 1.0f, 1.0f,
	 1.0f,  1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 1.0f, 0.0f,
	 1.0f, -1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f,
  
	// +y
	-1.0f,  1.0f, -1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
	-1.0f,  1.0f,  1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f,
	 1.0f,  1.0f,  1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
	 1.0f,  1.0f, -1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
  
	// -y
	-1.0f, -1.0f, -1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 1.0f,
	 1.0f, -1.0f, -1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 1.0f,
	 1.0f, -1.0f,  1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f,
	-1.0f, -1.0f,  1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f,
  
	// +x
	 1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
	 1.0f,  1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
	 1.0f,  1.0f,  1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
	 1.0f, -1.0f,  1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  
	// -x
	-1.0f, -1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
	-1.0f, -1.0f,  1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
	-1.0f,  1.0f,  1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
	-1.0f,  1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
};

static int boxVerticesSize = 192;

static unsigned short boxIndices[] = {
	0,  1,  2,      0,  2,  3,    // +z
	4,  5,  6,      4,  6,  7,    // -z
	8,  9,  10,     8,  10, 11,   // +y
	12, 13, 14,     12, 14, 15,   // -y
	16, 17, 18,     16, 18, 19,   // +x
	20, 21, 22,     20, 22, 23    // -x
};

static int boxIndicesSize = 36;


/**
 * <!--  ~MeshManager():  -->
 */
MeshManager::~MeshManager() {
	if( boxMeshData != nullptr ) {
		delete boxMeshData;
	}
	if( sphereMeshData != nullptr ) {
		delete sphereMeshData;
	}
}

/**
 * <!--  getBoxMesh():  -->
 */
const Mesh* MeshManager::getBoxMesh(Material* material) {
	if( boxMeshData == nullptr ) {
		MeshFaceData* meshFaceData = new MeshFaceData(boxVertices,
													 boxVerticesSize,
													 boxIndices,
													 boxIndicesSize);
		MeshData* meshData = new MeshData();
		meshData->addMeshFace(meshFaceData, "");
		boxMeshData = meshData;
	}

	return boxMeshData->toMesh(material);
}

/**
 * <!--  getSphereMesh():  -->
 */
const Mesh* MeshManager::getSphereMesh(Material* material) {
	if( sphereMeshData == nullptr ) {
		const int rings = 20;
		const int sectors = 20;

		const float R = 1.0f / (float)(rings-1);
		const float S = 1.0f / (float)(sectors-1);

		int verticesSize = rings * sectors * 8;
		float* vertices = new float[verticesSize];
		float* v = vertices;
		
		for(int r=0; r<rings; ++r) {
			for(int s=0; s<sectors; ++s) {
				const float y = sin(-M_PI_2 + M_PI * r * R);
				const float x = cos(2*M_PI * s * S) * sin(M_PI * r * R);
				const float z = sin(2*M_PI * s * S) * sin(M_PI * r * R);

				*v++ = x; // vertex
				*v++ = y;
				*v++ = z;
				*v++ = x; // normal
				*v++ = y;
				*v++ = z;
				*v++ = s*S; // U
				*v++ = r*R; // V
			}
		}

		int indicesSize = (rings-1) * (sectors-1) * 6;
		unsigned short* indices = new unsigned short[indicesSize];
		unsigned short* ind = indices;
		
		for(int r=0; r<rings-1; ++r) {
			for(int s=0; s<sectors-1; ++s) {
				unsigned short index0 = r * sectors + s;
				unsigned short index1 = r * sectors + (s+1);
				unsigned short index2 = (r+1) * sectors + (s+1);
				unsigned short index3 = (r+1) * sectors + s;
				
				*ind++ = index0;
				*ind++ = index1;
				*ind++ = index2;
				*ind++ = index0;
				*ind++ = index2;
				*ind++ = index3;
			}
		}

		MeshFaceData* meshFaceData = new MeshFaceData(vertices,
													  verticesSize,
													  indices,
													  indicesSize);
		MeshData* meshData = new MeshData();
		meshData->addMeshFace(meshFaceData, "");
		sphereMeshData = meshData;
		
		delete [] vertices;
		delete [] indices;
	}

	return sphereMeshData->toMesh(material);
}

/**
 * <!--  getModelMesh():  -->
 */
const Mesh* MeshManager::getModelMesh(const char* path) {
	// TODO: 未実装
	return nullptr;
}
