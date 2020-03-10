// -*- C++ -*-
#ifndef MESHDATA_HEADER
#define MESHDATA_HEADER

#include <vector>
#include <string>
using namespace std;

class MeshFaceData;
class Material;
class Mesh;
class TextureManager;
class ShaderManager;


class CollisionMeshTriangle {
public:
    float x0;
    float y0;
    float z0;
    float x1;
    float y1;
    float z1;
    float x2;
    float y2;
    float z2;

    CollisionMeshTriangle(
        float x0_, float y0_, float z0_,
        float x1_, float y1_, float z1_,
        float x2_, float y2_, float z2_)
        :
        x0(x0_), y0(y0_), z0(z0_),
        x1(x1_), y1(y1_), z1(z1_),
        x2(x2_), y2(y2_), z2(z2_) {
    }
};


class CollisionMeshData {
private:
    vector<CollisionMeshTriangle> triangles;
    
public:
    void addCollisionTriangle(float x0, float y0, float z0,
                              float x1, float y1, float z1,
                              float x2, float y2, float z2) {
        triangles.push_back(CollisionMeshTriangle(x0, y0, z0,
                                                  x1, y1, z1,
                                                  x2, y2, z2));
    }

    const vector<CollisionMeshTriangle>& getTriangles() const {
        return triangles;
    }
};


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
