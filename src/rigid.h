// -*- C++ -*-
#ifndef RIGID_HEADER
#define RIGID_HEADER

#include "btBulletDynamicsCommon.h"
#include <math.h>

#ifndef M_PI_8
#define M_PI_8     0.5 * M_PI_4
#endif

#define NUM_LEGS 4
#define BODYPART_COUNT 2 * NUM_LEGS + 1
#define JOINT_COUNT BODYPART_COUNT - 1


class Rig {
private:
	btDynamicsWorld*	world;
	btCollisionShape*	shapes[BODYPART_COUNT];
	btRigidBody*		bodies[BODYPART_COUNT];
	btTypedConstraint*	joints[JOINT_COUNT];

	btRigidBody* localCreateRigidBody(btScalar mass,
									  const btTransform& startTransform,
									  btCollisionShape* shape);

public:
	Rig(btDynamicsWorld* world_, const btVector3& positionOffset);
	~Rig();

	btTypedConstraint** getJoints() {
		return &joints[0];
	}
};


class RigidManager {
	//keep the collision shapes, for deletion/cleanup
	btAlignedObjectArray<btCollisionShape*>	collisionShapes;
	btBroadphaseInterface* broadPhase;
	btCollisionDispatcher* dispatcher;
	btConstraintSolver*	solver;
	btDefaultCollisionConfiguration* configuration;
	btDiscreteDynamicsWorld* world;

	float time;
	float cyclePeriod; // in milliseconds
	float muscleStrength;
	
	btAlignedObjectArray<class Rig*> rigs;
	
public:
	RigidManager()
		:
		broadPhase(nullptr),
		dispatcher(nullptr),
		solver(nullptr),
		configuration(nullptr),
		world(nullptr) {
	}

	~RigidManager() {
	}

	void stepSimulation(float deltaTime);


	btRigidBody* createRigidBody(float mass,
								 const btTransform& startTransform,
								 btCollisionShape* shape,
								 const btVector4& color = btVector4(1, 0, 0, 1));

	void initPhysics();
	void exitPhysics();
	
	void spawnRig(const btVector3& startOffset);
	void setMotorTargets(btScalar deltaTime);
};

#endif
