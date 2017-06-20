#include "Environment.h"
#include <stdio.h>
#include <GLUT/glut.h>

#include "OffscreenRenderer.h"
#include "ScreenRenderer.h"


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

static btRigidBody* createRigidBody(btScalar mass,
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
	return body;
}

Model::Model(btDynamicsWorld* world_,
			 const btVector3& positionOffset)
	:
	world(world_) {
	
	// Setup the geometry
	shapes[BODYPART_PELVIS]          = new btCapsuleShapeZ(btScalar(0.05), btScalar(0.20));
	shapes[BODYPART_SPINE]           = new btCapsuleShapeZ(btScalar(0.15), btScalar(0.28));
	shapes[BODYPART_HEAD]            = new btCapsuleShapeZ(btScalar(0.10), btScalar(0.05));
	shapes[BODYPART_LEFT_UPPER_LEG]  = new btCapsuleShape(btScalar(0.07), btScalar(0.45));
	shapes[BODYPART_LEFT_LOWER_LEG]  = new btCapsuleShape(btScalar(0.05), btScalar(0.37));
	shapes[BODYPART_RIGHT_UPPER_LEG] = new btCapsuleShape(btScalar(0.07), btScalar(0.45));
	shapes[BODYPART_RIGHT_LOWER_LEG] = new btCapsuleShape(btScalar(0.05), btScalar(0.37));
	shapes[BODYPART_LEFT_UPPER_ARM]  = new btCapsuleShape(btScalar(0.05), btScalar(0.33));
	shapes[BODYPART_LEFT_LOWER_ARM]  = new btCapsuleShape(btScalar(0.04), btScalar(0.25));
	shapes[BODYPART_RIGHT_UPPER_ARM] = new btCapsuleShape(btScalar(0.05), btScalar(0.33));
	shapes[BODYPART_RIGHT_LOWER_ARM] = new btCapsuleShape(btScalar(0.04), btScalar(0.25));

	// Setup all the rigid bodies
	btTransform offset; offset.setIdentity();
	offset.setOrigin(positionOffset);

	// 骨盤
	btTransform transform;
	transform.setIdentity();
	transform.setOrigin(btVector3(btScalar(0.0), btScalar(1.0), btScalar(0.0)));
	bodies[BODYPART_PELVIS] = createRigidBody(btScalar(1.), offset*transform, shapes[BODYPART_PELVIS]);

	// 脊柱
	transform.setIdentity();
	transform.setOrigin(btVector3(btScalar(0.0), btScalar(1.0), btScalar(-0.2)));
	bodies[BODYPART_SPINE] = createRigidBody(btScalar(1.), offset*transform, shapes[BODYPART_SPINE]);

	// 頭
	transform.setIdentity();
	transform.setOrigin(btVector3(btScalar(0.0), btScalar(1.0), btScalar(-0.6)));
	bodies[BODYPART_HEAD] = createRigidBody(btScalar(1.), offset*transform, shapes[BODYPART_HEAD]);

	// 左上足
	transform.setIdentity();
	transform.setOrigin(btVector3(btScalar(-0.18), btScalar(0.65+0.175-0.1), btScalar(0.125)));
	bodies[BODYPART_LEFT_UPPER_LEG] = createRigidBody(btScalar(1.), offset*transform, shapes[BODYPART_LEFT_UPPER_LEG]);

	// 左下足
	transform.setIdentity();
	transform.setOrigin(btVector3(btScalar(-0.18), btScalar(0.2+0.175-0.1), btScalar(0.125)));
	bodies[BODYPART_LEFT_LOWER_LEG] = createRigidBody(btScalar(1.), offset*transform, shapes[BODYPART_LEFT_LOWER_LEG]);

	// 右上足
	transform.setIdentity();
	transform.setOrigin(btVector3(btScalar(0.18), btScalar(0.65+0.175-0.1), btScalar(0.125)));
	bodies[BODYPART_RIGHT_UPPER_LEG] = createRigidBody(btScalar(1.), offset*transform, shapes[BODYPART_RIGHT_UPPER_LEG]);

	// 右下足
	transform.setIdentity();
	transform.setOrigin(btVector3(btScalar(0.18), btScalar(0.2+0.175-0.1), btScalar(0.125)));
	bodies[BODYPART_RIGHT_LOWER_LEG] = createRigidBody(btScalar(1.), offset*transform, shapes[BODYPART_RIGHT_LOWER_LEG]);

	// 左上腕
	transform.setIdentity();
	//transform.setOrigin(btVector3(btScalar(-0.35), btScalar(1.0), btScalar(-0.45)));
	transform.setOrigin(btVector3(btScalar(-0.18), btScalar(1.0-0.17), btScalar(-0.45)));
	//transform.getBasis().setEulerZYX(0,0,M_PI_2);
	bodies[BODYPART_LEFT_UPPER_ARM] = createRigidBody(btScalar(1.), offset*transform, shapes[BODYPART_LEFT_UPPER_ARM]);

	// 左下腕
	transform.setIdentity();
	//transform.setOrigin(btVector3(btScalar(-0.7), btScalar(1.0), btScalar(-0.45)));
	transform.setOrigin(btVector3(btScalar(-0.18), btScalar(1.0-0.52), btScalar(-0.45)));
	//transform.getBasis().setEulerZYX(0,0,M_PI_2);
	bodies[BODYPART_LEFT_LOWER_ARM] = createRigidBody(btScalar(1.), offset*transform, shapes[BODYPART_LEFT_LOWER_ARM]);

	// 右上腕
	transform.setIdentity();
	//transform.setOrigin(btVector3(btScalar(0.35), btScalar(1.0), btScalar(-0.45)));
	transform.setOrigin(btVector3(btScalar(0.18), btScalar(1.0-0.17), btScalar(-0.45)));
	//transform.getBasis().setEulerZYX(0,0,-M_PI_2);
	bodies[BODYPART_RIGHT_UPPER_ARM] = createRigidBody(btScalar(1.), offset*transform, shapes[BODYPART_RIGHT_UPPER_ARM]);

	// 右下腕
	transform.setIdentity();
	//transform.setOrigin(btVector3(btScalar(0.7), btScalar(1.0), btScalar(-0.45)));
	transform.setOrigin(btVector3(btScalar(0.18), btScalar(1.0-0.52), btScalar(-0.45)));
	//transform.getBasis().setEulerZYX(0,0,-M_PI_2);
	bodies[BODYPART_RIGHT_LOWER_ARM] = createRigidBody(btScalar(1.), offset*transform, shapes[BODYPART_RIGHT_LOWER_ARM]);

	world->addRigidBody(bodies[BODYPART_PELVIS]);
	world->addRigidBody(bodies[BODYPART_SPINE]);
	world->addRigidBody(bodies[BODYPART_HEAD]);
	world->addRigidBody(bodies[BODYPART_LEFT_UPPER_LEG]);
	world->addRigidBody(bodies[BODYPART_LEFT_LOWER_LEG]);
	world->addRigidBody(bodies[BODYPART_RIGHT_UPPER_LEG]);
	world->addRigidBody(bodies[BODYPART_RIGHT_LOWER_LEG]);
	world->addRigidBody(bodies[BODYPART_LEFT_UPPER_ARM]);
	world->addRigidBody(bodies[BODYPART_LEFT_LOWER_ARM]);
	world->addRigidBody(bodies[BODYPART_RIGHT_UPPER_ARM]);
	world->addRigidBody(bodies[BODYPART_RIGHT_LOWER_ARM]);

	// Setup some damping on the bodies
	for (int i = 0; i < BODYPART_COUNT; ++i) {
		bodies[i]->setDamping(btScalar(0.05), btScalar(0.85));
		bodies[i]->setDeactivationTime(btScalar(0.8));
		bodies[i]->setSleepingThresholds(btScalar(1.6), btScalar(2.5));
	}

	// Now setup the constraints
	btHingeConstraint* hingeC;
	btConeTwistConstraint* coneC;

	btTransform localA, localB;

	// 骨盤->脊柱
	localA.setIdentity(); localB.setIdentity();
	// (もともと正面向きに)
	//localA.getBasis().setEulerZYX(0,M_PI_2,0); localA.setOrigin(btVector3(btScalar(0.0), btScalar(0.0), btScalar(-0.15)));
	//localB.getBasis().setEulerZYX(0,M_PI_2,0); localB.setOrigin(btVector3(btScalar(0.0), btScalar(0.0), btScalar(0.15)));
	localA.getBasis().setEulerZYX(0,0,M_PI_2); localA.setOrigin(btVector3(btScalar(0.0), btScalar(0.0), btScalar(-0.15)));
	localB.getBasis().setEulerZYX(0,0,M_PI_2); localB.setOrigin(btVector3(btScalar(0.0), btScalar(0.0), btScalar(0.15)));
	
	hingeC =  new btHingeConstraint(*bodies[BODYPART_PELVIS], *bodies[BODYPART_SPINE], localA, localB);
	//hingeC->setLimit(btScalar(-M_PI_4), btScalar(M_PI_2));
	hingeC->setLimit(btScalar(0.0), btScalar(M_PI_2));
	joints[JOINT_PELVIS_SPINE] = hingeC;
	world->addConstraint(joints[JOINT_PELVIS_SPINE], true);

	// 脊柱->頭
	localA.setIdentity(); localB.setIdentity();
	// (もともと上向きに)
	localA.getBasis().setEulerZYX(0,M_PI_2,0); localA.setOrigin(btVector3(btScalar(0.), btScalar(0.0), btScalar(-0.30)));
	localB.getBasis().setEulerZYX(0,M_PI_2,0); localB.setOrigin(btVector3(btScalar(0.), btScalar(0.0), btScalar(0.14)));
	coneC = new btConeTwistConstraint(*bodies[BODYPART_SPINE], *bodies[BODYPART_HEAD], localA, localB);
	coneC->setLimit(M_PI_4, M_PI_4, M_PI_2);
	joints[JOINT_SPINE_HEAD] = coneC;
	world->addConstraint(joints[JOINT_SPINE_HEAD], true);

	// 骨盤->左上足
	localA.setIdentity(); localB.setIdentity();
	localA.getBasis().setEulerZYX(0,0,-M_PI_4*5); localA.setOrigin(btVector3(btScalar(-0.18), btScalar(0.0), btScalar(0.1)));
	localB.getBasis().setEulerZYX(0,0,-M_PI_4*5); localB.setOrigin(btVector3(btScalar(0.), btScalar(0.225), btScalar(0.0)));
	coneC = new btConeTwistConstraint(*bodies[BODYPART_PELVIS], *bodies[BODYPART_LEFT_UPPER_LEG], localA, localB);
	coneC->setLimit(M_PI_4, M_PI_4, 0);
	joints[JOINT_LEFT_HIP] = coneC;
	world->addConstraint(joints[JOINT_LEFT_HIP], true);

	// 左上足->左下足
	localA.setIdentity(); localB.setIdentity();
	localA.getBasis().setEulerZYX(0,M_PI_2,0); localA.setOrigin(btVector3(btScalar(0.), btScalar(-0.225), btScalar(0.)));
	localB.getBasis().setEulerZYX(0,M_PI_2,0); localB.setOrigin(btVector3(btScalar(0.), btScalar(0.185), btScalar(0.)));
	hingeC =  new btHingeConstraint(*bodies[BODYPART_LEFT_UPPER_LEG], *bodies[BODYPART_LEFT_LOWER_LEG], localA, localB);
	hingeC->setLimit(btScalar(0), btScalar(M_PI * 0.9));
	joints[JOINT_LEFT_KNEE] = hingeC;
	world->addConstraint(joints[JOINT_LEFT_KNEE], true);

	// 骨盤->右上足
	localA.setIdentity(); localB.setIdentity();
	localA.getBasis().setEulerZYX(0,0,M_PI_4); localA.setOrigin(btVector3(btScalar(0.18), btScalar(0.0), btScalar(0.1)));
	localB.getBasis().setEulerZYX(0,0,M_PI_4); localB.setOrigin(btVector3(btScalar(0.), btScalar(0.225), btScalar(0.0)));
	coneC = new btConeTwistConstraint(*bodies[BODYPART_PELVIS], *bodies[BODYPART_RIGHT_UPPER_LEG], localA, localB);
	coneC->setLimit(M_PI_4, M_PI_4, 0);
	joints[JOINT_RIGHT_HIP] = coneC;
	world->addConstraint(joints[JOINT_RIGHT_HIP], true);

	// 右上足->右下足
	localA.setIdentity(); localB.setIdentity();
	localA.getBasis().setEulerZYX(0,M_PI_2,0); localA.setOrigin(btVector3(btScalar(0.), btScalar(-0.225), btScalar(0.)));
	localB.getBasis().setEulerZYX(0,M_PI_2,0); localB.setOrigin(btVector3(btScalar(0.), btScalar(0.185), btScalar(0.)));
	hingeC =  new btHingeConstraint(*bodies[BODYPART_RIGHT_UPPER_LEG], *bodies[BODYPART_RIGHT_LOWER_LEG], localA, localB);
	hingeC->setLimit(btScalar(0), btScalar(M_PI * 0.9));
	joints[JOINT_RIGHT_KNEE] = hingeC;
	world->addConstraint(joints[JOINT_RIGHT_KNEE], true);

	// 脊柱->左上腕
	localA.setIdentity(); localB.setIdentity();
	localA.getBasis().setEulerZYX(0,0,-M_PI_4*5); localA.setOrigin(btVector3(btScalar(-0.2), btScalar(0.0), btScalar(-0.15)));
	localB.getBasis().setEulerZYX(0,0,-M_PI_4*5); localB.setOrigin(btVector3(btScalar(0.), btScalar(0.18), btScalar(0.)));
	coneC = new btConeTwistConstraint(*bodies[BODYPART_SPINE], *bodies[BODYPART_LEFT_UPPER_ARM], localA, localB);
	coneC->setLimit(M_PI_4, M_PI_4, 0);
	joints[JOINT_LEFT_SHOULDER] = coneC;
	world->addConstraint(joints[JOINT_LEFT_SHOULDER], true);

	// 左上腕->左下腕
	localA.setIdentity(); localB.setIdentity();
	localA.getBasis().setEulerZYX(0,M_PI_2,0); localA.setOrigin(btVector3(btScalar(0.), btScalar(-0.18), btScalar(0.)));
	localB.getBasis().setEulerZYX(0,M_PI_2,0); localB.setOrigin(btVector3(btScalar(0.), btScalar(0.14), btScalar(0.)));
	hingeC =  new btHingeConstraint(*bodies[BODYPART_LEFT_UPPER_ARM], *bodies[BODYPART_LEFT_LOWER_ARM], localA, localB);
	hingeC->setLimit(btScalar(-M_PI_2), btScalar(0));
	joints[JOINT_LEFT_ELBOW] = hingeC;
	world->addConstraint(joints[JOINT_LEFT_ELBOW], true);


	// 脊柱->右上腕
	localA.setIdentity(); localB.setIdentity();
	localA.getBasis().setEulerZYX(0,0,M_PI_4); localA.setOrigin(btVector3(btScalar(0.2), btScalar(0.0), btScalar(-0.15)));
	localB.getBasis().setEulerZYX(0,0,M_PI_4); localB.setOrigin(btVector3(btScalar(0.), btScalar(0.18), btScalar(0.)));
	coneC = new btConeTwistConstraint(*bodies[BODYPART_SPINE], *bodies[BODYPART_RIGHT_UPPER_ARM], localA, localB);
	coneC->setLimit(M_PI_4, M_PI_4, 0);
	joints[JOINT_RIGHT_SHOULDER] = coneC;
	world->addConstraint(joints[JOINT_RIGHT_SHOULDER], true);

	// 右上腕->右下腕
	localA.setIdentity(); localB.setIdentity();
	localA.getBasis().setEulerZYX(0,M_PI_2,0); localA.setOrigin(btVector3(btScalar(0.), btScalar(-0.18), btScalar(0.)));
	localB.getBasis().setEulerZYX(0,M_PI_2,0); localB.setOrigin(btVector3(btScalar(0.), btScalar(0.14), btScalar(0.)));
	hingeC =  new btHingeConstraint(*bodies[BODYPART_RIGHT_UPPER_ARM], *bodies[BODYPART_RIGHT_LOWER_ARM], localA, localB);
	hingeC->setLimit(btScalar(-M_PI_2), btScalar(0));
	joints[JOINT_RIGHT_ELBOW] = hingeC;
	world->addConstraint(joints[JOINT_RIGHT_ELBOW], true);
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

	/*
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
	*/
}

void Environment::init() {
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

	auto motorPreTickCallback = [](btDynamicsWorld* world, btScalar timeStep) {
		Environment* manager = (Environment*)world->getWorldUserInfo();
		manager->setMotorTargets(timeStep);
	};

	world->setInternalTickCallback(motorPreTickCallback, this, true);

	//..
	world->setDebugDrawer(new DebugDrawer());
	int debugMode =
		btIDebugDraw::DBG_DrawWireframe |
		btIDebugDraw::DBG_DrawConstraints |
		btIDebugDraw::DBG_DrawConstraintLimits;
	world->getDebugDrawer()->setDebugMode(debugMode);
	//..
	
	// Setup a big ground box
	{
		btCollisionShape* groundShape = new btBoxShape(btVector3(btScalar(200.0),
																 btScalar(10.0),
																 btScalar(200.0)));
		collisionShapes.push_back(groundShape);
		btTransform groundTransform;
		groundTransform.setIdentity();
		groundTransform.setOrigin(btVector3(0,-10,0));
		btRigidBody* body = createRigidBody(btScalar(0.0), groundTransform, groundShape);
		world->addRigidBody(body);
	}

	//world->setGravity(btVector3(0,-10,0));
	world->setGravity(btVector3(0,-10,0));

	// Spawn one model
	const btVector3 startOffset(0, 0.5, 0);
	model = new Model(world, startOffset);
}

void Environment::release() {
	if( model != nullptr ) {
		delete model;
		model = nullptr;
	}

	// remove the rigidbodies from the dynamics world and delete them
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

	if( renderer != nullptr ) {
		renderer->release();
		delete renderer;
		renderer = nullptr;
	}
}

void Environment::setMotorTargets(btScalar deltaTime) {
	// deltaTimeの単位はsecond
	float deltaTimeUs = deltaTime * 1000000.0f; // microSec
	const float minDeltaTimeUs = 1000000.0f/60.f;

	if(deltaTimeUs > minDeltaTimeUs) {
		deltaTimeUs = minDeltaTimeUs;
	}

	timeUs += deltaTimeUs;

	// set per-frame sinusoidal position targets using angular motor
	if( model != nullptr ) {
		model->setMotorTargets(timeUs, deltaTimeUs);
	}
}

void Environment::step() {
	const float deltaTime = 1.0f/60.0f;

	if( renderer != nullptr ) {
		renderer->renderPre();
	}
	
	if(world) {
		world->stepSimulation(deltaTime);
		// Debug drawing
		world->debugDrawWorld();
	}

	if( renderer != nullptr ) {
		renderer->render();
	}
}

bool Environment::initRenderer(int width, int height, bool offscreen) {
	if( offscreen ) {
		renderer = new OffscreenRenderer();
	} else {
		renderer = new ScreenRenderer();
	}
	return renderer->init(width, height);
}

const void* Environment::getFrameBuffer() const {
	if( renderer != nullptr ) {
		return renderer->getBuffer();
	} else {
		return nullptr;
	}
}

int Environment::getFrameBufferWidth() const {
	if( renderer != nullptr ) {
		return renderer->getFrameBufferWidth();
	} else {
		return 0;
	}
}

int Environment::getFrameBufferHeight() const {
	if( renderer != nullptr ) {
		return renderer->getFrameBufferHeight();
	} else {
		return 0;
	}
}

int Environment::getFrameBufferSize() const {
	if( renderer != nullptr ) {
		return renderer->getFrameBufferSize();
	} else {
		return 0;
	}
}

void Environment::setRenderCamera(const Matrix4f& mat) {
	if( renderer != nullptr ) {
		renderer->setCamera(mat);
	}
}
