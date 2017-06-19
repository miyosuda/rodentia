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

btRigidBody* Model::createRigidBody(btScalar mass,
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

Model::Model(btDynamicsWorld* world_,
			 const btVector3& positionOffset)
	:
	world(world_) {
	
	const btVector3 vUp(0, 1, 0);
	const btVector3 vRight(1, 0, 0);

	// Setup geometry
	const float bodyWidth  = 0.3f;
	const float bodyHeight = 0.1f;
	const float bodyDepth  = 0.5f;
	const float legLength = 0.45f;     // つけねの長さ
	const float foreLegLength = 0.35f; // 足の先の長さ
	
	shapes[0] = new btBoxShape(btVector3(btScalar(bodyWidth),
										 btScalar(bodyHeight),
										 btScalar(bodyDepth)));
	
	for(int i=0; i<NUM_LEGS; ++i) {
		shapes[1 + 2*i] = new btCapsuleShape(btScalar(0.10),
											 btScalar(legLength));
		shapes[2 + 2*i] = new btCapsuleShape(btScalar(0.08), // 先の方が少し細い
											 btScalar(foreLegLength));
	}
	
	// Setup rigid bodies
	btTransform offset; offset.setIdentity();
	offset.setOrigin(positionOffset);
	
	// root
	const float height = 0.5;
	btVector3 vRoot = btVector3(btScalar(0.0), btScalar(height), btScalar(0.0));
	btTransform transform;
	transform.setIdentity();
	transform.setOrigin(vRoot);
	
	bodies[0] = createRigidBody(btScalar(5.0), //..
								offset * transform,
								shapes[0]);
	
	btVector3 vBoneOrigins[4];
	vBoneOrigins[0].setValue(-bodyWidth, 0.0f, -bodyDepth); // 左奥
	vBoneOrigins[1].setValue( bodyWidth, 0.0f, -bodyDepth); // 右奥
	vBoneOrigins[2].setValue(-bodyWidth, 0.0f,  bodyDepth); // 左手前
	vBoneOrigins[3].setValue( bodyWidth, 0.0f,  bodyDepth); // 右手前
	
	// legs
	for(int i=0; i<NUM_LEGS; ++i) {
		// thigh (太腿)		
		transform.setIdentity();
		transform.setOrigin( btVector3(vBoneOrigins[i].x(),
									   height,
									   vBoneOrigins[i].z() - 0.5 * legLength) );
		transform.setRotation(btQuaternion(vRight, M_PI_2));
		bodies[2*i+1] = createRigidBody(btScalar(1.0),
										offset * transform,
										shapes[2*i+1]);
		
		// shin (すね)
		transform.setIdentity();
		transform.setOrigin( btVector3(vBoneOrigins[i].x(),
									   height,
									   vBoneOrigins[i].z() - legLength - 0.5 * foreLegLength));
		transform.setRotation(btQuaternion(vRight, M_PI_2));
		bodies[2*i+2] = createRigidBody(btScalar(0.5), //..
										offset * transform,
										shapes[2*i+2]);
	}
	
	// Setup some damping on the bodies
	for(int i=0; i<BODYPART_COUNT; ++i) {
		bodies[i]->setDamping(0.05, 0.85);
		bodies[i]->setDeactivationTime(0.8);
		bodies[i]->setSleepingThresholds(0.5f, 0.5f);
	}

	// Setup the constraints
	btHingeConstraint* hingeC;
	
	btTransform localA, localB, localC;
	
	for(int i=0; i<NUM_LEGS; ++i) {
		// hip joints (足の付け根)
		localA.setIdentity();
		localB.setIdentity();
		localA.getBasis().setEulerZYX(0, -M_PI_2, 0); // Y軸(垂直軸)での回転 //..
		// 親側での位置
		localA.setOrigin(vBoneOrigins[i]);
		// 子側での位置
		localB = bodies[2*i+1]->getWorldTransform().inverse() *
			bodies[0]->getWorldTransform() * localA;
		hingeC = new btHingeConstraint(*bodies[0], *bodies[2*i+1],
									   localA, localB);
		hingeC->setLimit(btScalar(M_PI * -0.8f), btScalar(M_PI * -0.6f));

		joints[2*i] = hingeC;
		world->addConstraint(joints[2*i], true);

		// knee joints (ヒザ)
		localA.setIdentity();
		localB.setIdentity();
		localC.setIdentity();
		localA.getBasis().setEulerZYX(0, -M_PI_2, 0); // Y軸(垂直軸)での回転
		// A座標系でのヒザの位置
		btVector3 vTmp;
		vTmp.setValue( vBoneOrigins[i].x(),
					   0.0f,
					   vBoneOrigins[i].z() - legLength );
		localA.setOrigin(vTmp);
		// B座標系でのヒザの位置
		localB = bodies[2*i+1]->getWorldTransform().inverse() *
			bodies[0]->getWorldTransform() * localA;
		// C座標系でのヒザの位置
		localC = bodies[2*i+2]->getWorldTransform().inverse() *
			bodies[0]->getWorldTransform() * localA;
		hingeC = new btHingeConstraint(*bodies[2*i+1], *bodies[2*i+2],
									   localB, localC);

		if( i < 2 ) {
			// 前足
			hingeC->setLimit(btScalar(M_PI * 0.2), btScalar(M_PI * 0.5));
		} else {
			// 後ろ足
			hingeC->setLimit(btScalar(M_PI * 0.2), btScalar(M_PI * 0.5));
		}
		joints[2*i+1] = hingeC;
		world->addConstraint(joints[2*i+1], true);
	}
}

Model::~Model() {
	// Remove all constraints
	for(int i = 0; i < JOINT_COUNT; ++i) {
		world->removeConstraint(joints[i]);
		delete joints[i]; joints[i] = nullptr;
	}

	// Remove all bodies and shapes
	for(int i = 0; i < BODYPART_COUNT; ++i) {
		world->removeRigidBody(bodies[i]);
		
		delete bodies[i]->getMotionState();
		
		delete bodies[i]; bodies[i] = nullptr;
		delete shapes[i]; shapes[i] = nullptr;
	}
}

void Model::setMotorTargets(float timeUs, float deltaTimeUs) {
	const float cyclePeriodMs = 2000.0f; // in milliSec
	// new SIMD solver for joints clips accumulated impulse, so the new limits for the motor
	// should be (numberOfsolverIterations * oldLimits)
	// currently solver uses 10 iterations, so:
	const float muscleStrength = 0.5f;
	
	btScalar targetRate =
		(int(timeUs / 1000) % int(cyclePeriodMs)) / cyclePeriodMs;
	btScalar targetAngleRate = 0.5 * (1 + sin(2 * M_PI * targetRate));

	for(int i=0; i<2*NUM_LEGS; i++) {
		btHingeConstraint* hingeC = static_cast<btHingeConstraint*>(getJoint(i));
		btScalar curAngle = hingeC->getHingeAngle();
		btScalar targetAngle =
			hingeC->getLowerLimit() + targetAngleRate *
			(hingeC->getUpperLimit() - hingeC->getLowerLimit());
		btScalar angleError  = targetAngle - curAngle;
		btScalar desiredAngularVel = 1000000.0f * angleError/deltaTimeUs;
		hingeC->enableAngularMotor(true, desiredAngularVel, muscleStrength);
	}
}

static void motorPreTickCallback(btDynamicsWorld* world, btScalar timeStep) {
	RigidManager* manager = (RigidManager*)world->getWorldUserInfo();
	manager->setMotorTargets(timeStep);
}

void RigidManager::initPhysics() {
	// Setup the basic world
	timeUs = 0;
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
		createRigidBody(btScalar(0.0), groundTransform,groundShape);
	}

	// Spawn one ragdoll
	btVector3 startOffset(0, 0.5, 0);
	spawnModel(startOffset);
}

void RigidManager::exitPhysics() {
	for(int i=0; i<models.size(); ++i) {
		Model* model = models[i];
		delete model;
	}

	//cleanup in the reverse order of creation/initialization

	//remove the rigidbodies from the dynamics world and delete them
	
	for(int i=world->getNumCollisionObjects()-1; i>=0 ; i--) {
		btCollisionObject* obj = world->getCollisionObjectArray()[i];
		btRigidBody* body = btRigidBody::upcast(obj);
		if(body && body->getMotionState()) {
			delete body->getMotionState();
		}
		world->removeCollisionObject( obj );
		delete obj;
	}

	//delete collision shapes
	for(int j=0; j<collisionShapes.size(); ++j) {
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

void RigidManager::spawnModel(const btVector3& startOffset) {
	Model* model = new Model(world, startOffset);
	models.push_back(model);
}

void RigidManager::setMotorTargets(btScalar deltaTime) {
	// deltaTimeの単位はsecond
	float deltaTimeUs = deltaTime * 1000000.0f; // microSec
	const float minDeltaTimeUs = 1000000.0f/60.f;

	if(deltaTimeUs > minDeltaTimeUs) {
		deltaTimeUs = minDeltaTimeUs;
	}

	timeUs += deltaTimeUs;

	// set per-frame sinusoidal position targets using angular motor
	for(int r=0; r<models.size(); r++) {
		models[r]->setMotorTargets(timeUs, deltaTimeUs);
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
										   btCollisionShape* shape) {
	
	btVector3 localInertia(0, 0, 0);

	bool isDynamic = (mass != 0.0f);
	if(isDynamic) {
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
