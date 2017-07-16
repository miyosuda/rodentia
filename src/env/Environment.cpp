#include "Environment.h"
#include <stdio.h>

#include "OffscreenRenderer.h"
#include "ScreenRenderer.h"
#include "DebugDrawer.h"

static const int ID_IGNORE_COLLISION = -1;
static const int ID_AGENT = -2;

static void convertBtTransformToMatrix4f(const btTransform& transform, Matrix4f& mat) {
	const btVector3& origin = transform.getOrigin();
	const btMatrix3x3& basis = transform.getBasis();
	for(int i=0; i<3; ++i) {
		btVector3 column = basis.getColumn(i);
		mat.setColumn(i, Vector4f(column.x(), column.y(), column.z(), 0.0f));
	}
	mat.setColumn(3, Vector4f(origin.x(), origin.y(), origin.z(), 1.0f));
}

//---------------------------
//   [RigidBodyComponent]
//---------------------------

RigidBodyComponent::RigidBodyComponent(float mass,
									   float posX, float posY, float posZ,
									   float rot,
									   btCollisionShape* shape,
									   btDynamicsWorld* world_,
									   int collisionId)
	:
	world(world_) {
	
	btTransform transform;
	transform.setIdentity();
	transform.setOrigin(btVector3(posX, posY, posZ));
	transform.getBasis().setEulerZYX(0.0f, rot, 0.0f);

	btVector3 localInertia(0,0,0);

	bool isDynamic = (mass != 0.0f);
	if (isDynamic) {
		shape->calculateLocalInertia(mass, localInertia);
	}

	btDefaultMotionState* motionState = new btDefaultMotionState(transform);
	btRigidBody::btRigidBodyConstructionInfo info(mass,
												  motionState,
												  shape,
												  localInertia);
	body = new btRigidBody(info);
	world->addRigidBody(body);
	
	body->setUserIndex(collisionId);
}

RigidBodyComponent::~RigidBodyComponent() {
	if(body->getMotionState()) {
		delete body->getMotionState();
	}
	world->removeCollisionObject(body);
	delete body;
}

int RigidBodyComponent::getCollisionId() const {
	return body->getUserIndex();
}

void RigidBodyComponent::control(const Action& action) {
}

void RigidBodyComponent::getMat(Matrix4f& mat) const {
	convertBtTransformToMatrix4f(body->getWorldTransform(), mat);
}


//---------------------------
// [AgentRigidBodyComponent]
//---------------------------
AgentRigidBodyComponent::AgentRigidBodyComponent(float mass,
												 float posX, float posY, float posZ,
												 float rot,
												 btCollisionShape* shape,
												 btDynamicsWorld* world_,
												 int collisionId)
	:
	RigidBodyComponent(mass,
					   posX,  posY,  posZ,
					   rot,
					   shape,
					   world_,
					   collisionId) {

	// Disable deactivation
	body->setActivationState(DISABLE_DEACTIVATION);
	
	// Set damping
	body->setDamping(btScalar(0.05), btScalar(0.85));

	// Set stand-up constraint
	// TODO: Agent can't move vertically with this constraint setting
	/*
	btTransform frameInA, frameInB;
	frameInA = btTransform::getIdentity();
	frameInB = btTransform::getIdentity();	
	frameInA.setOrigin(btVector3(0.0, 10.0, 0.0));
	frameInB.setOrigin(btVector3(0.0, -1.0, 0.0));
	btGeneric6DofConstraint* constraint =
		new btGeneric6DofConstraint(*floorBody, *body,
									frameInA, frameInB,
									true);

	constraint->setLinearLowerLimit(btVector3(-SIMD_INFINITY, 0, -SIMD_INFINITY));
	constraint->setLinearUpperLimit(btVector3( SIMD_INFINITY, 0,  SIMD_INFINITY));
	world->addConstraint(constraint);
	*/
}

void AgentRigidBodyComponent::control(const Action& action) {
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

//---------------------------
//   [EnvironmentObject]
//---------------------------
EnvironmentObject::EnvironmentObject() {
}

EnvironmentObject::~EnvironmentObject() {
	delete rigidBodyComponent;
}

//---------------------------
//      [StageObject]
//---------------------------
StageObject::StageObject(float posX, float posY, float posZ,
						 float rot,
						 btCollisionShape* shape,
						 btDynamicsWorld* world,
						 int collisionId)
	:   
	EnvironmentObject() {
	rigidBodyComponent = new RigidBodyComponent(0.0f,
												posX, posY, posZ,
												rot,
												shape,
												world,
												collisionId);
}

//---------------------------
//      [AgentObject]
//---------------------------
AgentObject::AgentObject(btCollisionShape* shape,
						 btDynamicsWorld* world,
						 int collisionId)
	:
	EnvironmentObject() {
	rigidBodyComponent = new AgentRigidBodyComponent(1.0f,
													 0.0f, 1.0, 0.0f,
													 0.0f,
													 shape,
													 world,
													 collisionId);
}

void AgentObject::control(const Action& action) {
	rigidBodyComponent->control(action);
}

void AgentObject::getMat(Matrix4f& mat) const {
	rigidBodyComponent->getMat(mat);
}

//---------------------------
//  [CollisionShapeManager]
//---------------------------

CollisionShapeManager::~CollisionShapeManager() {
	// Delete collision shapes
	for(int j=0; j<collisionShapes.size(); ++j) {
		btCollisionShape* shape = collisionShapes[j];
		delete shape;
	}
}

btCollisionShape* CollisionShapeManager::getSphereShape(float radius) {
	// TODO: same shape should be cached and reused
	btCollisionShape* shape = new btSphereShape(radius);
	collisionShapes.push_back(shape);
	return shape;
}

btCollisionShape* CollisionShapeManager::getBoxShape(float halfExtentX,
													 float halfExtentY,
													 float halfExtentZ) {
	// TODO: same shape should be cached and reused
	btCollisionShape* shape = new btBoxShape(btVector3(halfExtentX,
													   halfExtentY,
													   halfExtentZ));
	collisionShapes.push_back(shape);
	return shape;
}

btCollisionShape* CollisionShapeManager::getCylinderShape(float halfExtentX,
														  float halfExtentY,
														  float halfExtentZ) {
	// TODO: same shape should be cached and reused
	btCollisionShape* shape = new btCylinderShape(btVector3(halfExtentX,
															halfExtentY,
															halfExtentZ));
	collisionShapes.push_back(shape);
	return shape;
}

//---------------------------
//      [Environment]
//---------------------------

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

	nextObjId = 0;

	// Add floor stage object
	addBox(200.0f, 10.0f, 200.0f,
		   0.0f, -10.0f, 0.0f,
		   0.0f,
		   false);	

	world->setGravity(btVector3(0, -10, 0));

	// Add agent object
	prepareAgent();
}

void Environment::prepareAgent() {
	// TOOD: constraint用にfloorを渡すようにしないといけない
	btCollisionShape* shape = collisionShapeManager.getCylinderShape(1.0f, 1.0f, 1.0f);
	agent = new AgentObject(shape, world, ID_AGENT);
}

void Environment::release() {
	if( agent != nullptr ) {
		delete agent;
		agent = nullptr;
	}

	for(auto itr=objectMap.begin(); itr!=objectMap.end(); ++itr) {
		auto object = itr->second;
		delete object;
	}
	objectMap.clear();
	
	// Delete debug drawer
	if( debugDrawer != nullptr ) {
		delete debugDrawer;
		debugDrawer = nullptr;
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

	nextObjId = 0;
}

void Environment::checkCollision() {
	collidedIds.clear();
	
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
				if( otherId != ID_AGENT && otherId != ID_IGNORE_COLLISION ) {
					collidedIds.push_back(otherId);
				}
			} else if( obj1->getUserIndex() == ID_AGENT ) {
				int otherId = obj0->getUserIndex();
				if( otherId != ID_AGENT && otherId != ID_IGNORE_COLLISION ) {
					collidedIds.push_back(otherId);
				}
			}
		}
	}
}

void Environment::step(const Action& action, bool updateCamera) {
	const float deltaTime = 1.0f/60.0f;

	if( renderer != nullptr ) {
		renderer->renderPre();
	}
	
	if(world) {
		if( agent != nullptr ) {
			agent->control(action);
		}
		
		world->stepSimulation(deltaTime);

		if( updateCamera ) {
			// TODO: updateCamera周りちゃんと整理すること
			updateCameraToAgentView();
		}

		// Collision check
		checkCollision();
		
		// Debug drawing
		if( debugDrawer != nullptr && renderer != nullptr ) {
			const Camera& camera = renderer->getCamera();
			debugDrawer->prepare(camera.getInvMat(),
								 camera.getProjectionMat());
			world->debugDrawWorld();
		}
	}

	if( renderer != nullptr ) {
		renderer->render();
	}
}

int Environment::addBox(float halfExtentX, float halfExtentY, float halfExtentZ,
						float posX, float posY, float posZ,
						float rot,
						bool detectCollision) {
	btCollisionShape* shape = collisionShapeManager.getBoxShape(halfExtentX,
																halfExtentY,
																halfExtentZ);
	return addObject(shape, posX, posY, posZ, rot, detectCollision);
}

int Environment::addSphere(float radius,
						   float posX, float posY, float posZ,
						   float rot,
						   bool detectCollision) {
	btCollisionShape* shape = collisionShapeManager.getSphereShape(radius);
	return addObject(shape, posX, posY, posZ, rot, detectCollision);
}

int Environment::addObject(btCollisionShape* shape,
						   float posX, float posY, float posZ,
						   float rot,
						   bool detectCollision) {
	int id = nextObjId;
	nextObjId += 1;

	int collisionId;

	if( detectCollision ) {
		collisionId = id;
	} else {
		collisionId = ID_IGNORE_COLLISION;
	}

	EnvironmentObject* object = new StageObject(
		posX, posY, posZ,
		rot,
		shape,
		world,
		collisionId);

	objectMap[id] = object;
	return id;
}

void Environment::removeObj(int id) {
	auto itr = objectMap.find(id);
	if( itr != objectMap.end() ) {
		EnvironmentObject* object = objectMap[id];
		delete object;
		objectMap.erase(itr);
	}
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
	bool ret = renderer->init(width, height);
	
	if( ret ) {
		Shader* lineShader = shaderManager.getShader("line");

		// Set debug drawer
		debugDrawer = new DebugDrawer(lineShader);
		world->setDebugDrawer(debugDrawer);
		int debugMode =
			btIDebugDraw::DBG_DrawWireframe |
			btIDebugDraw::DBG_DrawConstraints |
			btIDebugDraw::DBG_DrawConstraintLimits;
		debugDrawer->setDebugMode(debugMode);
	}

	return ret;
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
		renderer->setCameraMat(mat);
	}
}

void Environment::updateCameraToAgentView() {
	if( agent != nullptr ) {
		Matrix4f mat;
		agent->getMat(mat);
		setRenderCamera(mat);
	}
}
