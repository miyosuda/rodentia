#include "RigidBodyComponent.h"
#include "Action.h"
#include "Matrix4f.h"

static void convertRigidBodyTransformToDrawMatrix4f(const btTransform& transform,
													const Vector3f& relativeCenter,
													Matrix4f& drawMat) {

	Matrix4f rigidBodyMat;
	const btVector3& origin = transform.getOrigin();
	const btMatrix3x3& basis = transform.getBasis();
	for(int i=0; i<3; ++i) {
		btVector3 column = basis.getColumn(i);
		rigidBodyMat.setColumn(i, Vector4f(column.x(), column.y(), column.z(), 0.0f));
	}
	rigidBodyMat.setColumn(3, Vector4f(origin.x(), origin.y(), origin.z(), 1.0f));

	Matrix4f invRelativeCenterMat;
	invRelativeCenterMat.setIdentity();
	invRelativeCenterMat.setColumn(3, Vector4f(-relativeCenter.x,
											   -relativeCenter.y,
											   -relativeCenter.z,
											   1.0f));

	drawMat.mul(rigidBodyMat, invRelativeCenterMat);
}

//---------------------------
//   [RigidBodyComponent]
//---------------------------

RigidBodyComponent::RigidBodyComponent(float mass,
									   const Vector3f& pos,
									   float rot,
									   const Vector3f& relativeCenter_,
									   btCollisionShape* shape,
									   btDynamicsWorld* world_,
									   int collisionId)
	:
	world(world_),
	relativeCenter(relativeCenter_) {

	btTransform drawTransform;
	drawTransform.setIdentity();
	drawTransform.setOrigin(btVector3(pos.x, pos.y, pos.z));
	drawTransform.getBasis().setEulerZYX(0.0f, rot, 0.0f);

	btTransform relativeCenterTransform;
	relativeCenterTransform.setIdentity();
	relativeCenterTransform.setOrigin(btVector3(relativeCenter.x,
												relativeCenter.y,
												relativeCenter.z));

	btTransform rigidBodyTransform;
	rigidBodyTransform = drawTransform * relativeCenterTransform;

	btVector3 localInertia(0,0,0);

	bool isDynamic = (mass != 0.0f);
	if (isDynamic) {
		shape->calculateLocalInertia(mass, localInertia);
	}

	btDefaultMotionState* motionState = new btDefaultMotionState(rigidBodyTransform);
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
	convertRigidBodyTransformToDrawMatrix4f(body->getWorldTransform(), relativeCenter, mat);
}

void RigidBodyComponent::getVeclocity(Vector3f& velocity) const {
	const btVector3& v = body->getLinearVelocity();
	velocity.set(v.x(), v.y(), v.z());
}

void RigidBodyComponent::locate(const Vector3f& pos, float rot) {
	btTransform drawTransform;
	drawTransform.setIdentity();
	drawTransform.setOrigin(btVector3(pos.x, pos.y, pos.z));
	drawTransform.getBasis().setEulerZYX(0.0f, rot, 0.0f);

	btTransform relativeCenterTransform;
	relativeCenterTransform.setIdentity();
	relativeCenterTransform.setOrigin(btVector3(relativeCenter.x,
												relativeCenter.y,
												relativeCenter.z));

	btTransform rigidBodyTransform;
	rigidBodyTransform = drawTransform * relativeCenterTransform;	
	
	body->setWorldTransform(rigidBodyTransform);
}


//---------------------------
// [AgentRigidBodyComponent]
//---------------------------
AgentRigidBodyComponent::AgentRigidBodyComponent(float mass,
												 const Vector3f& pos,
												 float rot,
												 btCollisionShape* shape,
												 btDynamicsWorld* world_,
												 int collisionId)
	:
	RigidBodyComponent(mass,
					   pos,
					   rot,
					   Vector3f(0.0f, 0.0f, 0.0f),
					   shape,
					   world_,
					   collisionId) {

	// Disable deactivation
	body->setActivationState(DISABLE_DEACTIVATION);
	
	// Set damping
	body->setDamping(btScalar(0.05), btScalar(0.85));

	// Disable rotaion around x,z axis 
	body->setAngularFactor(btVector3(0.0f, 1.0f, 0.0f));
}

void AgentRigidBodyComponent::control(const Action& action) {
	const float linearVelocityRate = 15.0f;
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
		targetLocalVelocity += btVector3(0.0f,
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
