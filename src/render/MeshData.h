// -*- C++ -*-
#ifndef MESHDATA_HEADER
#define MESHDATA_HEADER

#include <vector>
#include <string>
using namespace std;

#include "CollisionMeshData.h"

class MeshFaceData;
class Material;
class Mesh;
class TextureManager;
class ShaderManager;


class MeshData {
private:
    vector<MeshFaceData*> meshFaceDatas;
    vector<string> texturePathes;
    CollisionMeshData collisionMeshData;

public:
    ~MeshData();
    void addMeshFace(MeshFaceData* meshFaceData, const string& texturePath);

    Mesh* toMesh(Material* material);
    Mesh* toMesh(TextureManager& textureManager, ShaderManager& shaderManager);

    void addCollisionTriangle(float x0, float y0, float z0,
                              float x1, float y1, float z1,
                              float x2, float y2, float z2) {
        collisionMeshData.addCollisionTriangle(x0, y0, z0,
                                               x1, y1, z1,
                                               x2, y2, z2);
    }
    const CollisionMeshData& getCollisionMeshData() const {
        return collisionMeshData;
    }
};

#endif
