// -*- C++ -*-
#ifndef ENVIRONMENTOBJECT_HEADER
#define ENVIRONMENTOBJECT_HEADER

#include "btBulletDynamicsCommon.h"

#include "RigidBodyComponent.h"

class RigidBodyComponent;
class DrawComponent;
class Action;
class Matrix4f;
class Vector3f;
class Camera;
class Mesh;

class EnvironmentObject {
protected:
	RigidBodyComponent* rigidBodyComponent;
	DrawComponent* drawComponent;

public:
	EnvironmentObject();
	virtual ~EnvironmentObject();
	int getCollisionId() const;
	void getMat(Matrix4f& mat) const;
	void draw(const Camera& camera) const;
	btRigidBody* getRigidBody() {
		return rigidBodyComponent->getRigidBody();
	}
};

class StageObject : public EnvironmentObject {
public:
	StageObject(float posX, float posY, float posZ,
				float rot,
				btCollisionShape* shape,
				btDynamicsWorld* world,
				int collisionId,
				const Mesh* mesh,
				const Vector3f& scale);
};

class AgentObject : public EnvironmentObject {
public:	
	AgentObject(btCollisionShape* shape,
				btDynamicsWorld* world,
				btRigidBody* floorBody,
				int collisionId);
	void control(const Action& action);
};

#endif
