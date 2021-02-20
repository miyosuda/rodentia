#include "MeshManager.h"

#include <math.h>

#include "MeshFace.h"
#include "MeshData.h"
#include "MeshFaceData.h"
#include "Mesh.h"
#include "Material.h"
#include "ObjImporter.h"
#include "TextureManager.h"
#include "ShaderManager.h"
#include "Vector3f.h"


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
    // +z (正面)
    -1.0f, -1.0f,  1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // left bottom
    1.0f, -1.0f,  1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, // right bottom
    1.0f,  1.0f,  1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, // right top
    -1.0f,  1.0f,  1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, // left top
  
    // -z (裏面)
    -1.0f, -1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 1.0f, 0.0f,
    -1.0f,  1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 1.0f, 1.0f,
    1.0f,  1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f,
    1.0f, -1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f,
  
    // +y (上面)
    -1.0f,  1.0f, -1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    -1.0f,  1.0f,  1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
    1.0f,  1.0f,  1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f,
    1.0f,  1.0f, -1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
  
    // -y (下面)
    -1.0f, -1.0f, -1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f,
    1.0f, -1.0f, -1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f,
    1.0f, -1.0f,  1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 1.0f,
    -1.0f, -1.0f,  1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 1.0f,
  
    // +x (右面)
    1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
    1.0f,  1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
    1.0f,  1.0f,  1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    1.0f, -1.0f,  1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  
    // -x (左面)
    -1.0f, -1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    -1.0f, -1.0f,  1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
    -1.0f,  1.0f,  1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
    -1.0f,  1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
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
    release();
}

/**
 * <!--  release():  -->
 */
void MeshManager::release() {
    for (auto itr=modelMeshDataMap.begin(); itr!=modelMeshDataMap.end(); ++itr) {
        MeshData* meshData = itr->second;
        delete meshData;
    }
    modelMeshDataMap.clear();
}

/**
 * <!--  getBoxMesh():  -->
 *
 * NOTE: Created box's size = (1,1,1).
 */
Mesh* MeshManager::getBoxMesh(Material* material,
                              const Vector3f& textureLoopSize) {
    // Find cached MeshData with dummy key string.
    char path[64];
    int ix = (int)textureLoopSize.x * 100.0f;
    int iy = (int)textureLoopSize.y * 100.0f;
    int iz = (int)textureLoopSize.z * 100.0f;
    sprintf(path, "primitive:box:%d,%d,%d", ix, iy, iz);

    auto itr = modelMeshDataMap.find(path);
    if( itr != modelMeshDataMap.end() ) {
        MeshData* meshData = itr->second;
        return meshData->toMesh(material);
    }

    float* vertices = new float[boxVerticesSize];

    // Change U,V for loop.
    for(int i=0; i<4 * 6; ++i) {
        float loopU = 1.0f;
        float loopV = 1.0f;

        if( (i/4) < 2 ) {
            // Z面
            loopU = textureLoopSize.x;
            loopV = textureLoopSize.y;
        } else if( (i/4) < 4 ) {
            // Y面
            loopU = textureLoopSize.z;
            loopV = textureLoopSize.x;
        } else {
            // X面
            loopU = textureLoopSize.z;
            loopV = textureLoopSize.y;
        }
        
        float vx = boxVertices[8*i + 0];
        float vy = boxVertices[8*i + 1];
        float vz = boxVertices[8*i + 2];
        float nx = boxVertices[8*i + 3];
        float ny = boxVertices[8*i + 4];
        float nz = boxVertices[8*i + 5];
        float u  = boxVertices[8*i + 6];
        float v  = boxVertices[8*i + 7];
        
        vertices[8*i + 0] = vx;
        vertices[8*i + 1] = vy;
        vertices[8*i + 2] = vz;
        vertices[8*i + 3] = nx;
        vertices[8*i + 4] = ny;
        vertices[8*i + 5] = nz;
        vertices[8*i + 6] = u * loopU;
        vertices[8*i + 7] = v * loopV;
    }
    
    MeshFaceData* meshFaceData = new MeshFaceData();
    bool ret = meshFaceData->init(vertices,
                                  boxVerticesSize,
                                  boxIndices,
                                  boxIndicesSize);

    delete [] vertices;
    
    if( !ret ) {
        delete meshFaceData;
        return nullptr;
    }
    
    MeshData* meshData = new MeshData();
    meshData->addMeshFace(meshFaceData, "");

    modelMeshDataMap[path] = meshData;
    
    return meshData->toMesh(material);
}

/**
 * <!--  getSphereMesh():  -->
 */
Mesh* MeshManager::getSphereMesh(Material* material) {
    // Find cached MeshData with dummy key string.
    const char* path = "primitive:sphere";
    
    auto itr = modelMeshDataMap.find(path);
    if( itr != modelMeshDataMap.end() ) {
        MeshData* meshData = itr->second;
        return meshData->toMesh(material);
    }
    
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
            *ind++ = index2;
            *ind++ = index1;
            *ind++ = index0;
            *ind++ = index3;                
            *ind++ = index2;
        }
    }

    MeshFaceData* meshFaceData = new MeshFaceData();

    bool ret = meshFaceData->init(vertices,
                                  verticesSize,
                                  indices,
                                  indicesSize);

    delete [] vertices;
    delete [] indices;
        
    if( !ret ) {
        delete meshFaceData;
        return nullptr;
    }
        
    MeshData* meshData = new MeshData();
    meshData->addMeshFace(meshFaceData, "");

    modelMeshDataMap[path] = meshData;
    
    return meshData->toMesh(material);
}

/**
 * <!--  getModelMesh():  -->
 */
Mesh* MeshManager::getModelMesh(const char* path,
                                TextureManager& textureManager,
                                Material* replacingMaterial,
                                ShaderManager& shaderManager) {
    auto itr = modelMeshDataMap.find(path);
    if( itr != modelMeshDataMap.end() ) {
        MeshData* meshData = itr->second;
        if( replacingMaterial != nullptr ) {
            return meshData->toMesh(replacingMaterial);
        } else {
            return meshData->toMesh(textureManager, shaderManager);
        }
    }
    
    MeshData* meshData = ObjImporter::import(path);
    if( meshData == nullptr ) {
        return nullptr;
    }

    modelMeshDataMap[path] = meshData;
    
    if( replacingMaterial != nullptr ) {
        return meshData->toMesh(replacingMaterial);
    } else {    
        return meshData->toMesh(textureManager, shaderManager);
    }
}

/**
 * <!--  getCollisionMeshData():  -->
 */
const CollisionMeshData* MeshManager::getCollisionMeshData(const char* path) const {
    auto itr = modelMeshDataMap.find(path);
    if( itr == modelMeshDataMap.end() ) {
        return nullptr;
    } else {
        const MeshData* meshData = itr->second;
        return &(meshData->getCollisionMeshData());
    }
}
