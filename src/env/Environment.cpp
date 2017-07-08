#include "Environment.h"
#include <stdio.h>

#include "OffscreenRenderer.h"
#include "ScreenRenderer.h"
#include "DebugDrawer.h"

static btRigidBody* createRigidBody(btScalar mass,
									const btTransform& startTransform,
									btCollisionShape* shape) {

	btVector3 localInertia(0,0,0);

	bool isDynamic = (mass != 0.0f);
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

Model::Model(btDynamicsWorld* world_)
	:
	world(world_) {
	
	shape = new btCylinderShape(btVector3(btScalar(1.0), btScalar(1.0), btScalar(1.0)));

	btTransform transform;
	transform.setIdentity();
	transform.setOrigin(btVector3(btScalar(0.0), btScalar(1.0), btScalar(0.0)));
	
	body = createRigidBody(btScalar(1.0), transform, shape);

	world->addRigidBody(body);

	body->setDamping(btScalar(0.05), btScalar(0.85));
	body->setDeactivationTime(btScalar(0.8));
	body->setSleepingThresholds(btScalar(1.6), btScalar(2.5));
}

Model::~Model() {
	world->removeRigidBody(body);
	
	delete body->getMotionState();
	
	delete body;
	body = nullptr;
	
	delete shape;
	shape = nullptr;
}

void Environment::init() {
	// Setup the basic world
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

	// Set debug drawer
	auto debugDrawer = new DebugDrawer();
	world->setDebugDrawer(debugDrawer);
	int debugMode =
		btIDebugDraw::DBG_DrawWireframe |
		btIDebugDraw::DBG_DrawConstraints |
		btIDebugDraw::DBG_DrawConstraintLimits;
	debugDrawer->setDebugMode(debugMode);
	
	// Setup a big ground box
	{
		btCollisionShape* shape = new btBoxShape(btVector3(btScalar(200.0),
														   btScalar(10.0),
														   btScalar(200.0)));
		collisionShapes.push_back(shape);
		btTransform transform;
		transform.setIdentity();
		transform.setOrigin(btVector3(0,-10,0));
		btRigidBody* body = createRigidBody(btScalar(0.0), transform, shape);
		world->addRigidBody(body);
	}

	world->setGravity(btVector3(0, -10, 0));

	model = new Model(world);
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

void Environment::step(const Action& action) {	
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
