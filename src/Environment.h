// -*- C++ -*-
#ifndef ENVIRONMENT_HEADER
#define ENVIRONMENT_HEADER

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

	btTypedConstraint* getJoint(int index) {
		return joints[index];
	}

public:
	Model(btDynamicsWorld* world_, const btVector3& positionOffset);
	~Model();

	void setMotorTargets(float timeUs, float deltaTimeUs);
};


class Environment {
	btAlignedObjectArray<btCollisionShape*>	collisionShapes;
	btBroadphaseInterface* broadPhase;
	btCollisionDispatcher* dispatcher;
	btConstraintSolver*	solver;
	btDefaultCollisionConfiguration* configuration;
	btDiscreteDynamicsWorld* world;

	float timeUs; // microSec
	Model* model;

	void setMotorTargets(btScalar deltaTime);
	
public:
	Environment()
		:
		broadPhase(nullptr),
		dispatcher(nullptr),
		solver(nullptr),
		configuration(nullptr),
		world(nullptr),
		model(nullptr) {
	}

	~Environment() {
	}

	void init();
	void release();
	void step();
};

#endif
