#include "EnvironmentObject.h"

#include "DrawComponent.h"
#include "Action.h"
#include "Matrix4f.h"
#include "Vector3f.h"
#include "Camera.h"
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

void EnvironmentObject::draw(const Camera& camera) const {
	if( drawComponent != nullptr ) {
		Matrix4f rigidBodyMat;
		getMat(rigidBodyMat);
		drawComponent->draw(camera, rigidBodyMat);
	}
}

//---------------------------
//      [StageObject]
//---------------------------
StageObject::StageObject(float posX, float posY, float posZ,
						 float rot,
						 btCollisionShape* shape,
						 btDynamicsWorld* world,
						 int collisionId,
						 const Mesh* mesh,
						 const Vector3f& scale)
	:
	EnvironmentObject() {
	rigidBodyComponent = new RigidBodyComponent(0.0f,
												posX, posY, posZ,
												rot,
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
						 btRigidBody* floorBody,
						 int collisionId)
	:
	EnvironmentObject() {
	rigidBodyComponent = new AgentRigidBodyComponent(1.0f,
													 0.0f, 1.0, 0.0f,
													 0.0f,
													 shape,
													 world,
													 floorBody,
													 collisionId);
}

void AgentObject::control(const Action& action) {
	rigidBodyComponent->control(action);
}

void AgentObject::locate(float posX, float posY, float posZ,
						 float rot) {
	rigidBodyComponent->locate(posX, posY, posZ, rot);
}
