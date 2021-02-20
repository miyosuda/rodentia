// -*- C++ -*-
#ifndef MESHMANAGER_HEADER
#define MESHMANAGER_HEADER

#include <map>
#include <string>

using namespace std;


class Mesh;
class MeshData;
class MeshFaceData;
class Material;
class TextureManager;
class ShaderManager;
class Vector3f;
class CollisionMeshData;


class MeshManager {
private:
    map<string, MeshData*> modelMeshDataMap;

public:
    MeshManager() {}
    ~MeshManager();
    void release();
    Mesh* getBoxMesh(Material* material,
                     const Vector3f& textureLoopSize);
    Mesh* getSphereMesh(Material* material);
    Mesh* getModelMesh(const char* path,
                       TextureManager& textureManager,
                       Material* replacingMaterial,
                       ShaderManager& shaderManager);
    const CollisionMeshData* getCollisionMeshData(const char* path) const;
};

#endif
