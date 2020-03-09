#include "MeshFaceData.h"
#include <string.h>
#include <float.h>

/**
 * <!--  MeshFaceData():  -->
 */
MeshFaceData::MeshFaceData() {
}

/**
 * <!--  init():  -->
 */
bool MeshFaceData::init( const float* vertices,
                         int verticesSize_,
                         const unsigned short* indices,
                         int indicesSize_ ) {
    verticesSize = verticesSize_;
    indicesSize = indicesSize_;

    for(int i=0; i<verticesSize/8; ++i) {
        float vx = vertices[8*i+0];
        float vy = vertices[8*i+1];
        float vz = vertices[8*i+2];
        boundingBox.mergeVertex(vx, vy, vz);
    }

    // Set GL buffer objects
    bool ret;
    
    ret = indexBuffer.init(indices, indicesSize);
    if(!ret) {
        printf("Failed to init IndexBuffer\n");
        return false;
    }

    ret = vertexBuffer.init(vertices, verticesSize);
    if(!ret) {
        return false;
    }

    // Rendering用のVAO設定
    ret = vertexArray.init();
    if(!ret) {
        printf("Failed to init VertexArray\n");
        return false;
    }
    vertexArray.bind();

    // このvertexBuffer.bind()は 後続のglVertexAttribPointer()の対象を指定する為に行っている.
    vertexBuffer.bind();
    
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4*8, (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 4*8, (void*)(4*3));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 4*8, (void*)(4*6));

    // このindexBuffer.bind() は VAOに対して行っている.
    indexBuffer.bind();

    vertexBuffer.unbind();

    // VAOのbind解除. indexBufferへのバインドは消える.
    vertexArray.unbind();

    // Shadow Depth用のVAO設定
    ret = depthVertexArray.init();
    if(!ret) {
        printf("Failed to init VertexArray\n");
        return false;
    }
    depthVertexArray.bind();

    vertexBuffer.bind();
    
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4*8, (void*)0);

    indexBuffer.bind();
    vertexBuffer.unbind();

    depthVertexArray.unbind();
    return true;
}

/**
 * <!--  release():  -->
 */
void MeshFaceData::release() {
    vertexBuffer.release();
    indexBuffer.release();
    vertexArray.release();

    depthVertexArray.release();
}

/**
 * <!--  draw():  -->
 */
void MeshFaceData::draw(bool forShadow) const {
    if( forShadow ) {
        depthVertexArray.bind();
    } else {
        vertexArray.bind();
    }
    
    glDrawElements( GL_TRIANGLES, indicesSize, GL_UNSIGNED_SHORT, 0 );

    if( forShadow ) {
        depthVertexArray.unbind();
    } else {
        vertexArray.unbind();
    }
}

/**
 * <!--  ~MeshFaceData():  -->
 */
MeshFaceData::~MeshFaceData() {
    release();
}
