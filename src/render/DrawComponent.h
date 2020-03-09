// -*- C++ -*-
#ifndef DRAWCOMPONENT_HEADER
#define DRAWCOMPONENT_HEADER

#include <vector>
using namespace std;

#include "Matrix4f.h"

class Vector3f;
class Mesh;
class RenderingContext;
class BoundingBox;
class Material;

class DrawComponent {
private:
    Matrix4f scaleMat;
    Mesh* mesh;

public:
    DrawComponent(Mesh* mesh_, const Vector3f& scale);
    ~DrawComponent();
    void draw(RenderingContext& context, const Matrix4f& rigidBodyMat) const;
    void calcBoundingBox(const Matrix4f& rigidBodyMat,
                         BoundingBox& boundingBox) const;
    void replaceMaterials(const vector<Material*>& materials);
};

#endif
