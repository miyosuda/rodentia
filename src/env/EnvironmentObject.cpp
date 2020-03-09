#include "EnvironmentObject.h"

#include "DrawComponent.h"
#include "Action.h"
#include "Vector3f.h"
#include "RenderingContext.h"
#include "Mesh.h"

//---------------------------
// [EnvironmentObjectInfo]
//---------------------------
void EnvironmentObjectInfo::set(const Matrix4f& mat, const Vector3f& velocity_) {
    velocity.set(velocity_);

    const Vector4f& trans = mat.getColumnRef(3);
    pos.set(trans.x, trans.y, trans.z);
    rot.set(mat);
}

//---------------------------
//   [EnvironmentObject]
//---------------------------
EnvironmentObject::EnvironmentObject()
    :
    drawComponent(nullptr) {
}

EnvironmentObject::~EnvironmentObject() {
    delete rigidBodyComponent;
    if( drawComponent != nullptr ) {
        delete drawComponent;
    }
}

// Get RigidBody Matrix (position and rotation)
void EnvironmentObject::getMat(Matrix4f& mat) const {
    rigidBodyComponent->getMat(mat);
}

void EnvironmentObject::draw(RenderingContext& context) const {
    if( drawComponent != nullptr ) {
        Matrix4f rigidBodyMat;
        getMat(rigidBodyMat);
        drawComponent->draw(context, rigidBodyMat);
    }
}

bool EnvironmentObject::calcBoundingBox(BoundingBox& boundingBox) {
    if( drawComponent != nullptr ) {
        Matrix4f rigidBodyMat;
        getMat(rigidBodyMat);
        drawComponent->calcBoundingBox(rigidBodyMat, boundingBox);
        return true;
    } else {
        return false;
    }
}

void EnvironmentObject::getInfo(EnvironmentObjectInfo& info) const {
    Matrix4f mat;
    getMat(mat);
    
    Vector3f velocity;
    rigidBodyComponent->getVeclocity(velocity);
    
    info.set(mat, velocity);
}

void EnvironmentObject::locate(const Vector3f& pos, const Quat4f& rot) {
    rigidBodyComponent->locate(pos, rot);
}

void EnvironmentObject::replaceMaterials(const vector<Material*>& materials) {
    if( drawComponent != nullptr ) {
        drawComponent->replaceMaterials(materials);
    }
}

//---------------------------
//      [StageObject]
//---------------------------
StageObject::StageObject(const Vector3f& pos,
                         const Quat4f& rot,
                         float mass,
                         const Vector3f& relativeCenter,
                         btCollisionShape* shape,
                         btDynamicsWorld* world,
                         int collisionId,
                         Mesh* mesh,
                         const Vector3f& scale)
    :
    EnvironmentObject() {
    rigidBodyComponent = new RigidBodyComponent(mass,
                                                pos,
                                                rot,
                                                relativeCenter,
                                                shape,
                                                world,
                                                collisionId);
    drawComponent = new DrawComponent(mesh, scale);
}

//---------------------------
//      [AgentObject]
//---------------------------
AgentObject::AgentObject(btCollisionShape* shape,
                         btDynamicsWorld* world,
                         int collisionId)
    :
    EnvironmentObject() {
    rigidBodyComponent = new AgentRigidBodyComponent(1.0f,
                                                     Vector3f(0.0f, 1.0, 0.0f),
                                                     0.0f,
                                                     shape,
                                                     world,
                                                     collisionId);
}

void AgentObject::control(const Action& action) {
    rigidBodyComponent->control(action);
}
