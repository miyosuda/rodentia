// -*- C++ -*-
#ifndef MESH_HEADER
#define MESH_HEADER

#include <vector>
using namespace std;

#include "BoundingBox.h"

class MeshFace;
class Matrix4f;
class RenderingContext;
class Material;


class Mesh {
private:
    vector<MeshFace*> meshFaces;
    BoundingBox boundingBox;

public:
    Mesh() {}
    ~Mesh();
    void addMeshFace(MeshFace* meshFace);
    void draw(const RenderingContext& context) const;
    const BoundingBox& getBoundingBox() const { return boundingBox; }

    void replaceMaterials(const vector<Material*>& materials);
};

#endif
