#include "MeshManager.h"
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
const Mesh* MeshManager::getBoxMesh(Texture* texture, Shader* shader) {
	if( boxMeshFaceData == nullptr ) {
		boxMeshFaceData = new MeshFaceData(boxVertices,
										   boxVerticesSize,
										   boxIndices,
										   boxIndicesSize);
	}

	// TODO: Materialを引数にした方がよいか？
	Material* material = new Material(texture, shader);
	MeshFace* meshFace = new MeshFace(material,
									  *boxMeshFaceData);
	Mesh* mesh = new Mesh();
	mesh->addMeshFace(meshFace);
	return mesh;
}
