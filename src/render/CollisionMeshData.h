// -*- C++ -*-
#ifndef COLLISIONMESHDATA_HEADER
#define COLLISIONMESHDATA_HEADER

#include <vector>
using namespace std;

#include "BoundingBox.h"


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
    BoundingBox boundingBox;
    
public:
    void addCollisionTriangle(float x0, float y0, float z0,
                              float x1, float y1, float z1,
                              float x2, float y2, float z2) {
        triangles.push_back(CollisionMeshTriangle(x0, y0, z0,
                                                  x1, y1, z1,
                                                  x2, y2, z2));
        boundingBox.mergeVertex(x0, y0, z0);
        boundingBox.mergeVertex(x1, y1, z1);
        boundingBox.mergeVertex(x2, y2, z2);
    }

    const vector<CollisionMeshTriangle>& getTriangles() const {
        return triangles;
    }

    const BoundingBox& getBoundingBox() const {
        return boundingBox;
    }
};

#endif
