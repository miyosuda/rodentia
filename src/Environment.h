// -*- C++ -*-
#ifndef ENVIRONMENT_HEADER
#define ENVIRONMENT_HEADER

#include "btBulletDynamicsCommon.h"
#include <math.h>

#ifndef M_PI_8
#define M_PI_8     0.5 * M_PI_4
#endif

//#define NUM_LEGS 4
//#define BODYPART_COUNT 2 * NUM_LEGS + 1
//#define JOINT_COUNT BODYPART_COUNT - 1

enum {
	BODYPART_PELVIS = 0,
	BODYPART_SPINE,
	BODYPART_HEAD,

	BODYPART_LEFT_UPPER_LEG,
	BODYPART_LEFT_LOWER_LEG,

	BODYPART_RIGHT_UPPER_LEG,
	BODYPART_RIGHT_LOWER_LEG,

	BODYPART_LEFT_UPPER_ARM,
	BODYPART_LEFT_LOWER_ARM,

	BODYPART_RIGHT_UPPER_ARM,
	BODYPART_RIGHT_LOWER_ARM,

	BODYPART_COUNT
};

enum {
	JOINT_PELVIS_SPINE = 0,
	JOINT_SPINE_HEAD,

	JOINT_LEFT_HIP,
	JOINT_LEFT_KNEE,

	JOINT_RIGHT_HIP,
	JOINT_RIGHT_KNEE,

	JOINT_LEFT_SHOULDER,
	JOINT_LEFT_ELBOW,

	JOINT_RIGHT_SHOULDER,
	JOINT_RIGHT_ELBOW,

	JOINT_COUNT
};


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
