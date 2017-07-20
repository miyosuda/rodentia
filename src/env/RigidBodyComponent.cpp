#include "RigidBodyComponent.h"
#include "Action.h"
#include "Matrix4f.h"

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

void RigidBodyComponent::locate(float posX, float posY, float posZ,
								float rot) {
	btTransform transform;
	transform.setIdentity();
	transform.setOrigin(btVector3(posX, posY, posZ));
	transform.getBasis().setEulerZYX(0.0f, rot, 0.0f);
	body->setWorldTransform(transform);
}


//---------------------------
// [AgentRigidBodyComponent]
//---------------------------
AgentRigidBodyComponent::AgentRigidBodyComponent(float mass,
												 float posX, float posY, float posZ,
												 float rot,
												 btCollisionShape* shape,
												 btDynamicsWorld* world_,
												 btRigidBody* floorBody,
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