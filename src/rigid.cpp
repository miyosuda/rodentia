#include "rigid.h"
#include <stdio.h>
#include <GLUT/glut.h>

class DebugDrawer: public btIDebugDraw {
private:
	int debugMode;
	
public:
	// debug mode functions
	virtual void setDebugMode(int debugMode_) override {
		debugMode = debugMode_;
	}
	virtual int getDebugMode() const override {
		return debugMode;
	}

	// drawing functions
	virtual void drawContactPoint(const btVector3 &pointOnB,
								  const btVector3 &normalOnB,
								  btScalar distance,
								  int lifeTime,
								  const btVector3 &color) override;
	virtual void drawLine(const btVector3 &from,
						  const btVector3 &to,
						  const btVector3 &color) override;

	// unused
	virtual void reportErrorWarning(const char* warningString) override {}
	virtual void draw3dText(const btVector3 &location, const char* textString) override {}
	
	void toggleDebugFlag(int flag);
};

void DebugDrawer::drawLine(const btVector3 &from,
						   const btVector3 &to,
						   const btVector3 &color) {

	// draws a simple line of pixels between points.
	
	// use the GL_LINES primitive to draw lines
	glBegin(GL_LINES);
	glColor3f(color.getX(), color.getY(), color.getZ());
	glVertex3f(from.getX(), from.getY(), from.getZ());
	glVertex3f(to.getX(), to.getY(), to.getZ());
	glEnd();
}

void DebugDrawer::drawContactPoint(const btVector3 &pointOnB,
								   const btVector3 &normalOnB,
								   btScalar distance,
								   int lifeTime,
								   const btVector3 &color) {
	// draws a line between two contact points
	btVector3 const startPoint = pointOnB;
	btVector3 const endPoint = pointOnB + normalOnB * distance;
	drawLine(startPoint, endPoint, color);
}

void DebugDrawer::toggleDebugFlag(int flag) {
	// checks if a flag is set and enables/
	// disables it
	if (debugMode & flag) {
		// flag is enabled, so disable it
		debugMode = debugMode & (~flag);
	} else {
		// flag is disabled, so enable it
		debugMode |= flag;
	}
}




btRigidBody* Rig::localCreateRigidBody(btScalar mass,
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

Rig::Rig(btDynamicsWorld* world_,
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

Rig::~Rig() {
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


void motorPreTickCallback(btDynamicsWorld *world, btScalar timeStep) {
	RigidManager* manager = (RigidManager*)world->getWorldUserInfo();
	manager->setMotorTargets(timeStep);
}

void RigidManager::initPhysics() {
	// Setup the basic world
	time = 0;
	cyclePeriod = 2000.0f; // in milliseconds

	// new SIMD solver for joints clips accumulated impulse, so the new limits for the motor
	// should be (numberOfsolverIterations * oldLimits)
	// currently solver uses 10 iterations, so:
	muscleStrength = 0.5f;

	configuration = new btDefaultCollisionConfiguration();

	dispatcher = new btCollisionDispatcher(configuration);

	btVector3 worldAabbMin(-10000,-10000,-10000);
	btVector3 worldAabbMax( 10000, 10000, 10000);
	broadPhase = new btAxisSweep3(worldAabbMin, worldAabbMax);

	solver = new btSequentialImpulseConstraintSolver;

	world = new btDiscreteDynamicsWorld(dispatcher,
										broadPhase,
										solver,
										configuration);

	world->setInternalTickCallback(motorPreTickCallback, this, true);

	world->setDebugDrawer(new DebugDrawer()); //..
	world->getDebugDrawer()->setDebugMode(true); //..
	
	// Setup a big ground box
	{
		btCollisionShape* groundShape = new btBoxShape(btVector3(btScalar(200.0),
																 btScalar(10.0),
																 btScalar(200.0)));
		collisionShapes.push_back(groundShape);
		btTransform groundTransform;
		groundTransform.setIdentity();
		groundTransform.setOrigin(btVector3(0,-10,0));
		createRigidBody(btScalar(0.),groundTransform,groundShape);
	}

	// Spawn one ragdoll
	btVector3 startOffset(0, 0.5, 0);
	spawnRig(startOffset);
}

void RigidManager::exitPhysics() {
	for(int i=0;i<rigs.size();i++) {
		Rig* rig = rigs[i];
		delete rig;
	}

	//cleanup in the reverse order of creation/initialization

	//remove the rigidbodies from the dynamics world and delete them
	
	for(int i=world->getNumCollisionObjects()-1; i>=0 ;i--) {
		btCollisionObject* obj = world->getCollisionObjectArray()[i];
		btRigidBody* body = btRigidBody::upcast(obj);
		if(body && body->getMotionState()) {
			delete body->getMotionState();
		}
		world->removeCollisionObject( obj );
		delete obj;
	}

	//delete collision shapes
	for(int j=0;j<collisionShapes.size();j++) {
		btCollisionShape* shape = collisionShapes[j];
		delete shape;
	}

	auto debugDrawer = world->getDebugDrawer();
	delete debugDrawer;

	delete world;
	delete solver;
	delete broadPhase;
	delete dispatcher;
	delete configuration;
}

void RigidManager::spawnRig(const btVector3& startOffset) {
	Rig* rig = new Rig(world, startOffset);
	rigs.push_back(rig);
}

void RigidManager::setMotorTargets(btScalar deltaTime) {
	float ms = deltaTime * 1000000.0f;
	float minFPS = 1000000.0f/60.f;
	if(ms > minFPS) {
		ms = minFPS;
	}

	time += ms;

	// set per-frame sinusoidal position targets using angular motor (hacky?)
	for(int r=0; r<rigs.size(); r++) {
		for(int i=0; i<2*NUM_LEGS; i++) {
			btHingeConstraint* hingeC =
				static_cast<btHingeConstraint*>(rigs[r]->getJoints()[i]);
			btScalar curAngle = hingeC->getHingeAngle();
			
			btScalar targetPercent =
				(int(time / 1000) % int(cyclePeriod)) / cyclePeriod;
			btScalar targetAngle   = 0.5 * (1 + sin(2 * M_PI * targetPercent));
			btScalar targetLimitAngle =
				hingeC->getLowerLimit() + targetAngle *
				(hingeC->getUpperLimit() - hingeC->getLowerLimit());
			btScalar angleError  = targetLimitAngle - curAngle;
			btScalar desiredAngularVel = 1000000.f * angleError/ms;
			hingeC->enableAngularMotor(true, desiredAngularVel, muscleStrength);
		}
	}
}


void RigidManager::stepSimulation(float deltaTime) {
	if(world) {
		world->stepSimulation(deltaTime);
		// Debug drawing
		world->debugDrawWorld();
	}
}

btRigidBody* RigidManager::createRigidBody(float mass,
										   const btTransform& startTransform,
										   btCollisionShape* shape,
										   const btVector4& color) {
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
