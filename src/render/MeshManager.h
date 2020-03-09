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
                       TextureManager& textureManager, ShaderManager& shaderManager);
};

#endif
