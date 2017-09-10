// -*- C++ -*-
#ifndef RIGIDBODYCOMPONENT_HEADER
#define RIGIDBODYCOMPONENT_HEADER

#include "btBulletDynamicsCommon.h"

#include "Vector3f.h"

class Matrix4f;
class Action;

class RigidBodyComponent {
protected:
	btDynamicsWorld* world;
	btRigidBody* body;
	Vector3f relativeCenter;

public:
	RigidBodyComponent(float mass,
					   const Vector3f& pos,
					   float rot,
					   const Vector3f& relativeCenter_,
					   btCollisionShape* shape,
					   btDynamicsWorld* world,
					   int collisionId);
	virtual ~RigidBodyComponent();
	int getCollisionId() const;
	virtual void control(const Action& action);
	void getMat(Matrix4f& mat) const;
	void locate(const Vector3f& pos, float rot);
	btRigidBody* getRigidBody() { return body; }

};

class AgentRigidBodyComponent : public RigidBodyComponent {
public:
	AgentRigidBodyComponent(float mass,
							const Vector3f& pos,
							float rot,
							btCollisionShape* shape,
							btDynamicsWorld* world,
							int collisionId);
	virtual void control(const Action& action) override;
	void getMat(Matrix4f& mat) const;
};

#endif
