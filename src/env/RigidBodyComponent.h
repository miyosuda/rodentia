// -*- C++ -*-
#ifndef RIGIDBODYCOMPONENT_HEADER
#define RIGIDBODYCOMPONENT_HEADER

#include "btBulletDynamicsCommon.h"

#include "Vector3f.h"
#include "Matrix4f.h"

class Action;
class EnvironmentObject;


class RigidBodyComponent {
protected:
    btDynamicsWorld* world;
    btRigidBody* body;
    Vector3f relativeCenter;

public:
    RigidBodyComponent(float mass,
                       const Vector3f& pos,
                       const Quat4f& rot,
                       const Vector3f& relativeCenter_,
                       btCollisionShape* shape,
                       btDynamicsWorld* world,
                       EnvironmentObject* obj);
    virtual ~RigidBodyComponent();
    virtual void control(const Action& action);
    virtual void getMat(Matrix4f& mat) const;
    void getVeclocity(Vector3f& velocity) const;
    virtual void locate(const Vector3f& pos, const Quat4f& rot);
    void applyImpulse(const Vector3f& impulse);
};

class AgentRigidBodyComponent : public RigidBodyComponent {
private:
    float rotY;
    
public:
    AgentRigidBodyComponent(float mass,
                            const Vector3f& pos,
                            float rotY_,
                            btCollisionShape* shape,
                            btDynamicsWorld* world,
                            EnvironmentObject* obj);
    virtual void control(const Action& action) override;
    void locate(const Vector3f& pos, const Quat4f& rot) override;
    void getMat(Matrix4f& mat) const override;
};

#endif
