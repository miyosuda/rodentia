// -*- C++ -*-
#ifndef RIGID_HEADER
#define RIGID_HEADER

#include "btBulletDynamicsCommon.h"

#ifndef M_PI
#define M_PI       3.14159265358979323846
#endif

#ifndef M_PI_2
#define M_PI_2     1.57079632679489661923
#endif

#ifndef M_PI_4
#define M_PI_4     0.785398163397448309616
#endif

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
									  btCollisionShape* shape) {

		btVector3 localInertia(0,0,0);

		bool isDynamic = (mass != 0.f);
		if (isDynamic) {
			shape->calculateLocalInertia(mass, localInertia);
		}

		btDefaultMotionState* motionState = new btDefaultMotionState(startTransform);
		btRigidBody::btRigidBodyConstructionInfo info(mass,
													  motionState,
													  shape,
													  localInertia);
		btRigidBody* body = new btRigidBody(info);

		world->addRigidBody(body);

		return body;
	}

public:
	Rig(btDynamicsWorld* world_,
		const btVector3& positionOffset)
		:
		world(world_) {
		
		const btVector3 vUp(0, 1, 0);

		// Setup geometry
		const float bodySize  = 0.25f;
		const float legLength = 0.45f;
		const float foreLegLength = 0.75f;
		
		shapes[0] = new btCapsuleShape(btScalar(bodySize), btScalar(0.10));
		
		for(int i=0; i<NUM_LEGS; ++i) {
			shapes[1 + 2*i] = new btCapsuleShape(btScalar(0.10),
												 btScalar(legLength));
			shapes[2 + 2*i] = new btCapsuleShape(btScalar(0.08),
												 btScalar(foreLegLength));
		}

		// Setup rigid bodies
		const float height = 0.5;
		btTransform offset; offset.setIdentity();
		offset.setOrigin(positionOffset);

		// root
		btVector3 vRoot = btVector3(btScalar(0.0), btScalar(height), btScalar(0.0));
		btTransform transform;
		transform.setIdentity();
		transform.setOrigin(vRoot);
		
		bodies[0] = localCreateRigidBody(btScalar(1.0),
										 offset * transform,
										 shapes[0]);
		
		// legs
		for(int i=0; i<NUM_LEGS; ++i) {
			float fAngle = 2 * M_PI * i / NUM_LEGS;
			float fSin = sin(fAngle);
			float fCos = cos(fAngle);

			transform.setIdentity();
			btVector3 vBoneOrigin = btVector3(btScalar(fCos*(bodySize+0.5*legLength)),
											  btScalar(height),
											  btScalar(fSin*(bodySize+0.5*legLength)));
			transform.setOrigin(vBoneOrigin);

			// thigh
			btVector3 vToBone = (vBoneOrigin - vRoot).normalize();
			btVector3 vAxis = vToBone.cross(vUp);			
			transform.setRotation(btQuaternion(vAxis, M_PI_2));
			bodies[1+2*i] = localCreateRigidBody(btScalar(1.),
												 offset*transform,
												 shapes[1+2*i]);

			// shin
			transform.setIdentity();
			transform.setOrigin(btVector3(btScalar(fCos*(bodySize+legLength)),
										  btScalar(height-0.5*foreLegLength),
										  btScalar(fSin*(bodySize+legLength))));
			bodies[2+2*i] = localCreateRigidBody(btScalar(1.0),
												 offset*transform,
												 shapes[2+2*i]);
		}

		// Setup some damping on the bodies
		for(int i = 0; i < BODYPART_COUNT; ++i) {
			bodies[i]->setDamping(0.05, 0.85);
			bodies[i]->setDeactivationTime(0.8);
			bodies[i]->setSleepingThresholds(0.5f, 0.5f);
		}

		// Setup the constraints
		btHingeConstraint* hingeC;

		btTransform localA, localB, localC;

		for(int i=0; i<NUM_LEGS; ++i) {
			float fAngle = 2 * M_PI * i / NUM_LEGS;
			float fSin = sin(fAngle);
			float fCos = cos(fAngle);

			// hip joints
			localA.setIdentity();
			localB.setIdentity();
			localA.getBasis().setEulerZYX(0,-fAngle,0);
			localA.setOrigin(btVector3(btScalar(fCos*bodySize),
									   btScalar(0.),
									   btScalar(fSin*bodySize)));
			localB = bodies[1+2*i]->getWorldTransform().inverse() *
				bodies[0]->getWorldTransform() * localA;
			hingeC = new btHingeConstraint(*bodies[0], *bodies[1+2*i],
										   localA, localB);
			hingeC->setLimit(btScalar(-0.75 * M_PI_4), btScalar(M_PI_8));

			joints[2*i] = hingeC;
			world->addConstraint(joints[2*i], true);

			// knee joints
			localA.setIdentity();
			localB.setIdentity();
			localC.setIdentity();
			localA.getBasis().setEulerZYX(0,-fAngle,0);
			localA.setOrigin(btVector3(btScalar(fCos*(bodySize+legLength)),
									   btScalar(0.0),
									   btScalar(fSin*(bodySize+legLength))));
			localB = bodies[1+2*i]->getWorldTransform().inverse() *
				bodies[0]->getWorldTransform() * localA;
			localC = bodies[2+2*i]->getWorldTransform().inverse() *
				bodies[0]->getWorldTransform() * localA;
			hingeC = new btHingeConstraint(*bodies[1+2*i], *bodies[2+2*i],
										   localB, localC);
			
			hingeC->setLimit(btScalar(-M_PI_8), btScalar(0.2));
			joints[1+2*i] = hingeC;
			world->addConstraint(joints[1+2*i], true);
		}
	}

	~Rig() {
		// Remove all constraints
		for(int i = 0; i < JOINT_COUNT; ++i) {
			world->removeConstraint(joints[i]);
			delete joints[i]; joints[i] = 0;
		}

		// Remove all bodies and shapes
		for(int i = 0; i < BODYPART_COUNT; ++i) {
			world->removeRigidBody(bodies[i]);
			
			delete bodies[i]->getMotionState();

			delete bodies[i]; bodies[i] = 0;
			delete shapes[i]; shapes[i] = 0;
		}
	}

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

	void stepSimulation(float deltaTime) {
		if(world) {
			world->stepSimulation(deltaTime);
			//..
			world->debugDrawWorld();
			//..
		}
	}

	btRigidBody* createRigidBody(float mass,
								 const btTransform& startTransform,
								 btCollisionShape* shape,
								 const btVector4& color = btVector4(1, 0, 0, 1)) {
		btAssert((!shape || shape->getShapeType() != INVALID_SHAPE_PROXYTYPE));

		//rigidbody is dynamic if and only if mass is non zero, otherwise static
		bool isDynamic =(mass != 0.f);

		btVector3 localInertia(0, 0, 0);
		if(isDynamic) {
			shape->calculateLocalInertia(mass, localInertia);
		}

		btDefaultMotionState* motionState = new btDefaultMotionState(startTransform);
		btRigidBody::btRigidBodyConstructionInfo info(mass,
													   motionState,
													   shape,
													   localInertia);

		btRigidBody* body = new btRigidBody(info);

		body->setUserIndex(-1);
		world->addRigidBody(body);
		return body;
	}

	void initPhysics();
	void exitPhysics();
	
	void spawnRig(const btVector3& startOffset);
	void setMotorTargets(btScalar deltaTime);
};

#endif
