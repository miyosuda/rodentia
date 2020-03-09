#include "DebugDrawer.h"
#include "glinc.h"
#include "Shader.h"
#include "Matrix4f.h"
#include "RenderingContext.h"


/**
 * <!--  init():  -->
 */
bool DebugDrawer::init() {
    bool ret;

    const unsigned short indices[2] = {
        0, 1
    };
    
    ret = indexBuffer.init(indices, 2);
    if(!ret) {
        return false;
    }
    
    ret = vertexArray.init();
    if(!ret) {
        return false;
    }

    const float vertices[12] = {
        0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, // x, y, z, r, g, b
        0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
    };
    ret = vertexBuffer.init(vertices, 12, true); // dyanmic=true
    if(!ret) {
        return false;
    }

    vertexBuffer.bind();
    
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4*6, (void*)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 4*6, (void*)(4*3));
    
    vertexBuffer.unbind();
    vertexArray.unbind();
    
    return true;
}

/**
 * <!--  drawLine():  -->
 */
void DebugDrawer::drawLine(const btVector3 &from,
                           const btVector3 &to,
                           const btVector3 &color) {

    // draws a simple line of pixels between points.
    
    float vertices[12];
    vertices[0] = from.x();
    vertices[1] = from.y();
    vertices[2] = from.z();

    vertices[3] = color.x();
    vertices[4] = color.y();
    vertices[5] = color.z();
    
    vertices[6] = to.x();
    vertices[7] = to.y();
    vertices[8] = to.z();

    vertices[9]  = color.x();
    vertices[10] = color.y();
    vertices[11] = color.z();

    vertexBuffer.modify(vertices, 12);

    lineShader->use();

    vertexArray.bind();
    indexBuffer.bind();
    
    glDrawElements(GL_LINES, 2, GL_UNSIGNED_SHORT, 0);
    
    vertexArray.unbind();

    indexBuffer.unbind();
}

/**
 * <!--  drawContactPoint():  -->
 */
void DebugDrawer::drawContactPoint(const btVector3 &pointOnB,
                                   const btVector3 &normalOnB,
                                   btScalar distance,
                                   int lifeTime,
                                   const btVector3 &color) {
    // draws a line between two contact points
    btVector3 const startPoint = pointOnB;
    btVector3 const endPoint = pointOnB + normalOnB * distance;
    drawLine(startPoint, endPoint, color);
}

/**
 * <!--  toggleDebugFlag():  -->
 */
void DebugDrawer::toggleDebugFlag(int flag) {
    // checks if a flag is set and enables/disables it
    if (debugMode & flag) {
        // flag is enabled, so disable it
        debugMode = debugMode & (~flag);
    } else {
        // flag is disabled, so enable it
        debugMode |= flag;
    }
}

/**
 * <!--  prepare():  -->
 */
void DebugDrawer::prepare(RenderingContext& context) {
    Matrix4f modelMat;
    modelMat.setIdentity();

    context.setModelMat(modelMat);
    lineShader->setup(context);
}
