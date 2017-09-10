#include "EnvironmentObject.h"

#include "DrawComponent.h"
#include "Action.h"
#include "Matrix4f.h"
#include "Vector3f.h"
#include "RenderingContext.h"
#include "Mesh.h"


//---------------------------
//   [EnvironmentObject]
//---------------------------
EnvironmentObject::EnvironmentObject()
	:
	drawComponent(nullptr) {
}

EnvironmentObject::~EnvironmentObject() {
	delete rigidBodyComponent;
	if( drawComponent != nullptr ) {
		delete drawComponent;
	}
}

// Get RigidBody Matrix (position and rotation)
void EnvironmentObject::getMat(Matrix4f& mat) const {
	rigidBodyComponent->getMat(mat);
}

void EnvironmentObject::draw(RenderingContext& context) const {
	if( drawComponent != nullptr ) {
		Matrix4f rigidBodyMat;
		getMat(rigidBodyMat);
		drawComponent->draw(context, rigidBodyMat);
	}
}

bool EnvironmentObject::calcBoundingBox(BoundingBox& boundingBox) {
	if( drawComponent != nullptr ) {
		Matrix4f rigidBodyMat;
		getMat(rigidBodyMat);
		drawComponent->calcBoundingBox(rigidBodyMat, boundingBox);
		return true;
	} else {
		return false;
	}
}

//---------------------------
//      [StageObject]
//---------------------------
StageObject::StageObject(const Vector3f& pos,
						 float rot,
						 const Vector3f& relativeCenter,
						 btCollisionShape* shape,
						 btDynamicsWorld* world,
						 int collisionId,
						 const Mesh* mesh,
						 const Vector3f& scale)
	:
	EnvironmentObject() {
	rigidBodyComponent = new RigidBodyComponent(0.0f,
												pos,
												rot,
												relativeCenter,
												shape,
												world,
												collisionId);
	drawComponent = new DrawComponent(mesh, scale);
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
													 Vector3f(0.0f, 1.0, 0.0f),
													 0.0f,
													 shape,
													 world,
													 collisionId);
}

void AgentObject::control(const Action& action) {
	rigidBodyComponent->control(action);
}

void AgentObject::locate(const Vector3f& pos,
						 float rot) {
	rigidBodyComponent->locate(pos, rot);
}
