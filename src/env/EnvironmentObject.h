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
class Mesh;
class RenderingContext;
class BoundingBox;


class EnvironmentObject {
protected:
	RigidBodyComponent* rigidBodyComponent;
	DrawComponent* drawComponent;

public:
	EnvironmentObject();
	virtual ~EnvironmentObject();
	int getCollisionId() const;
	void getMat(Matrix4f& mat) const;
	void draw(RenderingContext& context) const;
	btRigidBody* getRigidBody() {
		return rigidBodyComponent->getRigidBody();
	}
	bool calcBoundingBox(BoundingBox& boundingBox);
};


class StageObject : public EnvironmentObject {
public:
	StageObject(const Vector3f& pos,
				float rot,
				float mass,
				const Vector3f& relativeCenter,
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
				int collisionId);
	void control(const Action& action);
	void locate(const Vector3f& pos,
				float rot);
};

#endif
