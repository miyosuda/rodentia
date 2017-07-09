#include "Environment.h"
#include <stdio.h>

#include "OffscreenRenderer.h"
#include "ScreenRenderer.h"
#include "DebugDrawer.h"

static const int ID_AGENT = 0;
static const int ID_IGNORE_COLLISION = -1;
static const int ID_OBJ_START = 1;

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

	body->setUserIndex(ID_AGENT);

	body->setActivationState(DISABLE_DEACTIVATION);
	world->addRigidBody(body);

	// Set damping
	body->setDamping(btScalar(0.05), btScalar(0.85));
}

Model::~Model() {
	world->removeRigidBody(body);
	
	delete body->getMotionState();
	
	delete body;
	body = nullptr;
	
	delete shape;
	shape = nullptr;
}

void Model::control(const Action& action) {
	const float linearVelocityRate = 5.0f;
	const float angularVelocityRate = 0.5f;
	const float impulseLengthLimit = 1.0f;
	
	// Calc linear impulse
	btVector3 targetLocalVelocity = btVector3(0.0f, 0.0f, 0.0f);
	
	if( action.strafe != 0 ) {
		// left and right
		targetLocalVelocity += btVector3(-linearVelocityRate * action.strafe,
										 0.0f,
										 0.0f);
	}
	
	if( action.move != 0 ) {
		// forward and backward
		targetLocalVelocity += btVector3( 0.0f,
										  0.0f,
										  -linearVelocityRate * action.move);
	}

	btTransform transform(body->getWorldTransform());
	transform.setOrigin(btVector3(0,0,0));

	btVector3 targetVelocity = transform * targetLocalVelocity;
	btVector3 velocityDiff = targetVelocity - body->getLinearVelocity();

	btVector3 impulse = velocityDiff / body->getInvMass();
	float impulseLen = impulse.length();

	if(impulseLen > impulseLengthLimit) {
		// Avoid too big impulse
		impulse *= (impulseLengthLimit / impulseLen);
	}

	// Apply impulse at the botom of cylinder.
	body->applyImpulse(impulse, btVector3(0.0f, -1.0f, 0.0f));

	// Calc angular impulse
	btVector3 targetLocalAngularVelocity = btVector3(0.0f, 0.0f, 0.0f);
	
	if( action.look != 0 ) {
		targetLocalAngularVelocity = btVector3(0.0f,
											   action.look * angularVelocityRate,
											   0.0f);
	}

	btVector3 targetAngularVelocity = transform * targetLocalAngularVelocity;
	btVector3 angularVelocityDiff = targetAngularVelocity - body->getAngularVelocity();
	btMatrix3x3 inertiaTensorWorld = body->getInvInertiaTensorWorld().inverse();
	btVector3 torqueImpulse = inertiaTensorWorld * angularVelocityDiff;
	
	body->applyTorqueImpulse(torqueImpulse);
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
	
	// Setup a ground floor box
	{
		btCollisionShape* shape = new btBoxShape(btVector3(btScalar(200.0),
														   btScalar(10.0),
														   btScalar(200.0)));
		collisionShapes.push_back(shape);
		btTransform transform;
		transform.setIdentity();
		transform.setOrigin(btVector3(0,-10,0));
		btRigidBody* body = createRigidBody(btScalar(0.0), transform, shape);
		body->setUserIndex(ID_IGNORE_COLLISION);
		world->addRigidBody(body);
	}

	nextObjId = ID_OBJ_START;

	world->setGravity(btVector3(0, -10, 0));

	model = new Model(world);
}

void Environment::release() {
	if( model != nullptr ) {
		delete model;
		model = nullptr;
	}

	// Remove the rigidbodies from the dynamics world and delete them
	for(int i=world->getNumCollisionObjects()-1; i>=0 ; i--) {
		btCollisionObject* obj = world->getCollisionObjectArray()[i];
		btRigidBody* body = btRigidBody::upcast(obj);
		if(body && body->getMotionState()) {
			delete body->getMotionState();
		}
		world->removeCollisionObject( obj );
		delete obj;
	}

	// Delete collision shapes
	for(int j=0; j<collisionShapes.size(); ++j) {
		btCollisionShape* shape = collisionShapes[j];
		delete shape;
	}

	auto debugDrawer = world->getDebugDrawer();
	if( debugDrawer != nullptr ) {
		delete debugDrawer;
	}

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

	nextObjId = ID_OBJ_START;
}

void Environment::checkCollision() {
	int numManifolds = world->getDispatcher()->getNumManifolds();
	
	for(int i=0; i<numManifolds; ++i) {
		btPersistentManifold* contactManifold =
			world->getDispatcher()->getManifoldByIndexInternal(i);
		const btCollisionObject* obj0 = contactManifold->getBody0();
		const btCollisionObject* obj1 = contactManifold->getBody1();

		bool hasContact = false;

		int numContacts = contactManifold->getNumContacts();
        for(int j=0; j<numContacts; ++j) {
            btManifoldPoint& pt = contactManifold->getContactPoint(j);
            if (pt.getDistance() < 0.0f) {
				hasContact = true;
            }
		}

		if( hasContact ) {
			if( obj0->getUserIndex() == ID_AGENT ) {
				int otherId = obj1->getUserIndex();
				if( otherId >= ID_OBJ_START ) {
					printf("collided=%d\n", otherId);
				}
			} else if( obj1->getUserIndex() == ID_AGENT ) {
				int otherId = obj0->getUserIndex();
				if( otherId >= ID_OBJ_START ) {
					printf("collided=%d\n", otherId);
				}
			}
		}
	}
}

void Environment::step(const Action& action) {
	const float deltaTime = 1.0f/60.0f;

	if( renderer != nullptr ) {
		renderer->renderPre();
	}
	
	if(world) {
		model->control(action);
		
		world->stepSimulation(deltaTime);

		// Collision check
		checkCollision();
		
		// Debug drawing
		world->debugDrawWorld();
	}

	if( renderer != nullptr ) {
		renderer->render();
	}
}

int Environment::addBox(float halfExtentX, float halfExtentY, float halfExtentZ,
						float posX, float posY, float posZ,
						float rot,
						bool detectCollision) {
	btCollisionShape* shape = new btBoxShape(btVector3(halfExtentX,
													   halfExtentX,
													   halfExtentZ));
	collisionShapes.push_back(shape);
	btTransform transform;
	transform.setIdentity();
	transform.setOrigin(btVector3(posX, posY, posZ));
	transform.getBasis().setEulerZYX(0.0f, rot, 0.0f);

	btRigidBody* body = createRigidBody(0.0, transform, shape);
	world->addRigidBody(body);

	int id = nextObjId;
	nextObjId += 1;

	if( detectCollision ) {
		body->setUserIndex(id);
	} else {
		body->setUserIndex(ID_IGNORE_COLLISION);
	}

	// TODO: idのmap管理

	return id;
}

int Environment::addSphere(float radius,
						   float posX, float posY, float posZ,
						   float rot,
						   bool detectCollision) {

	btCollisionShape* shape = new btSphereShape(radius);
	collisionShapes.push_back(shape);
	
	btTransform transform;
	transform.setIdentity();
	transform.setOrigin(btVector3(posX, posY, posZ));
	transform.getBasis().setEulerZYX(0.0f, rot, 0.0f);
	
	btRigidBody* body = createRigidBody(0.0, transform, shape);
	world->addRigidBody(body);

	int id = nextObjId;
	nextObjId += 1;

	if( detectCollision ) {
		body->setUserIndex(id);
	} else {
		body->setUserIndex(ID_IGNORE_COLLISION);
	}

	// TODO: idのmap管理
	
	return id;
}

void Environment::removeObj(int id) {
	// TODO: Not implemented yet
}

void Environment::locateAgent(float posX, float posY, float posZ,
							  float rot) {
	// TODO: Not implemented yet
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
