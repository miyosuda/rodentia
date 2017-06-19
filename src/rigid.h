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


class Model {
private:
	btDynamicsWorld*	world;
	btCollisionShape*	shapes[BODYPART_COUNT];
	btRigidBody*		bodies[BODYPART_COUNT];
	btTypedConstraint*	joints[JOINT_COUNT];

public:
	Model(btDynamicsWorld* world_, const btVector3& positionOffset);
	~Model();

	void setMotorTargets(float timeUs, float deltaTimeUs);

	btTypedConstraint* getJoint(int index) {
		return joints[index];
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

	float timeUs; // microSec
	btAlignedObjectArray<class Model*> models;
	
	void spawnModel(const btVector3& startOffset);
	
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

	void initPhysics();
	void exitPhysics();
	void setMotorTargets(btScalar deltaTime);
	void stepSimulation(float deltaTime);
};

#endif
