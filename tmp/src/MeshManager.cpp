#include "MeshManager.h"

#include <math.h>

#include "MeshFace.h"
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

static short boxIndices[] = {
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
	if( boxMeshFaceData != nullptr ) {
		delete boxMeshFaceData;
	}
}

/**
 * <!--  getBoxMesh():  -->
 */
const Mesh* MeshManager::getBoxMesh(Material* material) {
	if( boxMeshFaceData == nullptr ) {
		boxMeshFaceData = new MeshFaceData(boxVertices,
										   boxVerticesSize,
										   boxIndices,
										   boxIndicesSize);
	}

	MeshFace* meshFace = new MeshFace(material,
									  *boxMeshFaceData);
	Mesh* mesh = new Mesh();
	mesh->addMeshFace(meshFace);
	return mesh;
}

/**
 * <!--  getSphereMesh():  -->
 */
const Mesh* MeshManager::getSphereMesh(Material* material) {
	if( sphereMeshFaceData == nullptr ) {
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

		int indicesSize = rings * sectors * 6;
		short* indices = new short[indicesSize];
		short* ind = indices;

		for(int r=0; r<rings-1; ++r) {
			for(int s=0; s<sectors-1; ++s) {
				short index0 = r * sectors + s;
				short index1 = r * sectors + (s+1);
				short index2 = (r+1) * sectors + (s+1);
				short index3 = (r+1) * sectors + s;
				
				*ind++ = index0;
				*ind++ = index1;
				*ind++ = index2;
				*ind++ = index0;
				*ind++ = index2;
				*ind++ = index3;
			}
		}

		sphereMeshFaceData = new MeshFaceData(vertices,
											  verticesSize,
											  indices,
											  indicesSize);
		
		delete [] vertices;
		delete [] indices;
	}

	MeshFace* meshFace = new MeshFace(material,
									  *sphereMeshFaceData);
	Mesh* mesh = new Mesh();
	mesh->addMeshFace(meshFace);
	return mesh;
}
