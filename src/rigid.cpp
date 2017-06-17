#include "rigid.h"
#include <stdio.h>
#include <GLUT/glut.h>

class DebugDrawer: public btIDebugDraw {
private:
	int debugMode;
	
public:
	// debug mode functions
	virtual void setDebugMode(int debugMode) override {
		this->debugMode = debugMode;
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
	virtual void reportErrorWarning(const char* warningString) override {
	}
	virtual void draw3dText(const btVector3 &location, const char* textString) override {
	}
	
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

	printf(">> drawContactPoint\n");
	/*
	// draws a line between two contact points
	btVector3 const startPoint = pointOnB;
	btVector3 const endPoint = pointOnB + normalOnB * distance;
	drawLine(startPoint, endPoint, color);
	*/
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


void motorPreTickCallback(btDynamicsWorld *world, btScalar timeStep) {
	RigidManager* motorDemo = (RigidManager*)world->getWorldUserInfo();
	motorDemo->setMotorTargets(timeStep);
}

void RigidManager::initPhysics() {
	// Setup the basic world
	time = 0;
	cyclePeriod = 2000.0f; // in milliseconds

	//	muscleStrength = 0.05f;
	// new SIMD solver for joints clips accumulated impulse, so the new limits for the motor
	// should be (numberOfsolverIterations * oldLimits)
	// currently solver uses 10 iterations, so:
	muscleStrength = 0.5f;

	configuration = new btDefaultCollisionConfiguration();

	dispatcher = new btCollisionDispatcher(configuration);

	btVector3 worldAabbMin(-10000,-10000,-10000);
	btVector3 worldAabbMax(10000,10000,10000);
	broadPhase = new btAxisSweep3(worldAabbMin, worldAabbMax);

	solver = new btSequentialImpulseConstraintSolver;

	world = new btDiscreteDynamicsWorld(dispatcher,
												  broadPhase,
												  solver,
												  configuration);

	world->setInternalTickCallback(motorPreTickCallback, this, true);

	//..
	world->setDebugDrawer(new DebugDrawer()); //.. TODO: testing
	world->getDebugDrawer()->setDebugMode(true); //..
	//..
	
	// Setup a big ground box
	{
		btCollisionShape* groundShape = new btBoxShape(btVector3(btScalar(200.),
																 btScalar(10.),
																 btScalar(200.)));
		collisionShapes.push_back(groundShape);
		btTransform groundTransform;
		groundTransform.setIdentity();
		groundTransform.setOrigin(btVector3(0,-10,0));
		createRigidBody(btScalar(0.),groundTransform,groundShape);
	}

	// Spawn one ragdoll
	btVector3 startOffset(1,0.5,0);
	spawnRig(startOffset, false);
	
	startOffset.setValue(-2,0.5,0);
	spawnRig(startOffset, true);
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

	delete world;
	delete solver;
	delete broadPhase;
	delete dispatcher;
	delete configuration;	
}

void RigidManager::spawnRig(const btVector3& startOffset, bool bFixed) {
	Rig* rig = new Rig(world, startOffset, bFixed);
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
