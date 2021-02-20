#include "RigidBodyComponent.h"
#include "Action.h"
#include <math.h>


//---------------------------
//   [RigidBodyComponent]
//---------------------------

RigidBodyComponent::RigidBodyComponent(float mass,
                                       const Vector3f& pos,
                                       const Quat4f& rot,
                                       const Vector3f& relativeCenter_,
                                       btCollisionShape* shape,
                                       btDynamicsWorld* world_,
                                       EnvironmentObject* obj)
    :
    world(world_),
    relativeCenter(relativeCenter_) {

    btTransform drawTransform;
    drawTransform.setIdentity();
    drawTransform.setOrigin(btVector3(pos.x, pos.y, pos.z));
    drawTransform.setRotation(btQuaternion(rot.x, rot.y, rot.z, rot.w));

    btTransform relativeCenterTransform;
    relativeCenterTransform.setIdentity();
    relativeCenterTransform.setOrigin(btVector3(relativeCenter.x,
                                                relativeCenter.y,
                                                relativeCenter.z));

    btTransform rigidBodyTransform;
    rigidBodyTransform = drawTransform * relativeCenterTransform;

    btVector3 localInertia(0,0,0);

    bool isDynamic = (mass != 0.0f);
    if (isDynamic) {
        shape->calculateLocalInertia(mass, localInertia);
    }

    btDefaultMotionState* motionState = new btDefaultMotionState(rigidBodyTransform);
    btRigidBody::btRigidBodyConstructionInfo info(mass,
                                                  motionState,
                                                  shape,
                                                  localInertia);
    body = new btRigidBody(info);
    world->addRigidBody(body);
    
    body->setUserPointer(obj);
}

RigidBodyComponent::~RigidBodyComponent() {
    if(body->getMotionState()) {
        delete body->getMotionState();
    }
    world->removeCollisionObject(body);
    delete body;
}

void RigidBodyComponent::control(const Action& action) {
}

void RigidBodyComponent::getMat(Matrix4f& mat) const {
    const btTransform& transform = body->getWorldTransform();
    
    Matrix4f rigidBodyMat;
    
    const btVector3& origin = transform.getOrigin();
    const btMatrix3x3& basis = transform.getBasis();
    for(int i=0; i<3; ++i) {
        btVector3 column = basis.getColumn(i);
        rigidBodyMat.setColumn(i, Vector4f(column.x(), column.y(), column.z(), 0.0f));
    }
    rigidBodyMat.setColumn(3, Vector4f(origin.x(), origin.y(), origin.z(), 1.0f));

    Matrix4f invRelativeCenterMat;
    invRelativeCenterMat.setIdentity();
    invRelativeCenterMat.setColumn(3, Vector4f(-relativeCenter.x,
                                               -relativeCenter.y,
                                               -relativeCenter.z,
                                               1.0f));

    mat.mul(rigidBodyMat, invRelativeCenterMat);
}

void RigidBodyComponent::getVeclocity(Vector3f& velocity) const {
    const btVector3& v = body->getLinearVelocity();
    velocity.set(v.x(), v.y(), v.z());
}

void RigidBodyComponent::locate(const Vector3f& pos, const Quat4f& rot) {
    btTransform drawTransform;
    drawTransform.setIdentity();
    drawTransform.setOrigin(btVector3(pos.x, pos.y, pos.z));
    drawTransform.getBasis().setRotation(btQuaternion(rot.x, rot.y, rot.z, rot.w));

    btTransform relativeCenterTransform;
    relativeCenterTransform.setIdentity();
    relativeCenterTransform.setOrigin(btVector3(relativeCenter.x,
                                                relativeCenter.y,
                                                relativeCenter.z));

    btTransform rigidBodyTransform;
    rigidBodyTransform = drawTransform * relativeCenterTransform;
    
    body->setWorldTransform(rigidBodyTransform);
}

void RigidBodyComponent::applyImpulse(const Vector3f& impulse) {
    // Apply impulse at the center of sphere.
    btVector3 impulse_(impulse.x, impulse.y, impulse.z);
    body->applyImpulse(impulse_, btVector3(0.0f, 0.0f, 0.0f));
}


//---------------------------
// [AgentRigidBodyComponent]
//---------------------------
AgentRigidBodyComponent::AgentRigidBodyComponent(float mass,
                                                 const Vector3f& pos,
                                                 float rotY_,
                                                 btCollisionShape* shape,
                                                 btDynamicsWorld* world_,
                                                 EnvironmentObject* obj)
    :
    RigidBodyComponent(mass,
                       pos,
                       Quat4f(0.0f, 0.0f, 0.0f, 1.0f), // Don't use rigidbody's rotation
                       Vector3f(0.0f, 0.0f, 0.0f),
                       shape,
                       world_,
                       obj) {
    rotY = rotY_;
    
    // Disable deactivation
    body->setActivationState(DISABLE_DEACTIVATION);
    
    // Disable rotaion around x,z axis 
    body->setAngularFactor(btVector3(0.0f, 1.0f, 0.0f));
}

void AgentRigidBodyComponent::control(const Action& action) {
    const float linearVelocityRate = 15.0f;
    const float impulseLengthLimit = 1.0f;
    
    // Calc linear impulse
    btVector3 targetLocalVelocity = btVector3(0.0f, 0.0f, 0.0f);
    
    if( action.strafe != 0 ) {
        // left and right
        targetLocalVelocity += btVector3(-linearVelocityRate * action.strafe,
                                         0.0f,
                                         0.0f);
    }
    
    if( action.move != 0 ) {
        // forward and backward
        targetLocalVelocity += btVector3(0.0f,
                                         0.0f,
                                         -linearVelocityRate * action.move);
    }
    
    btTransform transform;
    transform.setIdentity();
    transform.setRotation(btQuaternion(0.0f, sin(rotY * 0.5f), 0.0f, cos(rotY * 0.5f)));

    btVector3 targetVelocity = transform * targetLocalVelocity;
    btVector3 velocityDiff = targetVelocity - body->getLinearVelocity();

    btVector3 impulse = velocityDiff / body->getInvMass();
    float impulseLen = impulse.length();

    if(impulseLen > impulseLengthLimit) {
        // Avoid too big impulse
        impulse *= (impulseLengthLimit / impulseLen);
    }

    // Not to damp vertical falling of the agent.
    impulse[1] = 0.0f;

    // Apply impulse at the center of sphere.
    body->applyImpulse(impulse, btVector3(0.0f, 0.0f, 0.0f));

    // Apply rotation
    float deltaRotY = (action.look / 180.0) * M_PI;
    rotY += deltaRotY;
}

void AgentRigidBodyComponent::locate(const Vector3f& pos, const Quat4f& rot) {
    // rot is applied only to rotY, and not applied to rigidBody's transform
    rotY = atan2(rot.y, rot.w) * 2.0f;

    btTransform drawTransform;
    drawTransform.setIdentity();
    drawTransform.setOrigin(btVector3(pos.x, pos.y, pos.z));

    btTransform relativeCenterTransform;
    relativeCenterTransform.setIdentity();
    relativeCenterTransform.setOrigin(btVector3(relativeCenter.x,
                                                relativeCenter.y,
                                                relativeCenter.z));

    btTransform rigidBodyTransform;
    rigidBodyTransform = drawTransform * relativeCenterTransform;
    
    body->setWorldTransform(rigidBodyTransform);
}

void AgentRigidBodyComponent::getMat(Matrix4f& mat) const {
    const btTransform& transform = body->getWorldTransform();
    
    Matrix4f rigidBodyMat;
    
    const btVector3& origin = transform.getOrigin();
    // rotation is calculated with rotY and not with rididbody's rotation.
    rigidBodyMat.setRotationY(rotY);
    rigidBodyMat.setColumn(3, Vector4f(origin.x(), origin.y(), origin.z(), 1.0f));

    Matrix4f invRelativeCenterMat;
    invRelativeCenterMat.setIdentity();
    invRelativeCenterMat.setColumn(3, Vector4f(-relativeCenter.x,
                                               -relativeCenter.y,
                                               -relativeCenter.z,
                                               1.0f));
    
    mat.mul(rigidBodyMat, invRelativeCenterMat);
}
