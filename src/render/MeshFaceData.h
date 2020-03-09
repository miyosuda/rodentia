// -*- C++ -*-
#ifndef MESHFACEDATA_HEADER
#define MESHFACEDATA_HEADER

#include "Vector3f.h"
#include "BufferObjects.h"
#include "BoundingBox.h"


class MeshFaceData {
private:
    int verticesSize;
    int indicesSize;

    BoundingBox boundingBox;

    VertexArray vertexArray;
    VertexBuffer vertexBuffer;
    IndexBuffer indexBuffer;

    VertexArray depthVertexArray;

    void release();

public:
    MeshFaceData();
    bool init(const float* vertices,
              int verticesSize_,
              const unsigned short* indices,
              int indicesSize_ );
    ~MeshFaceData();
    void draw(bool forShadow) const;
    const BoundingBox& getBoundingBox() const { return boundingBox; }
};

#endif
