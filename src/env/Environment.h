// -*- C++ -*-
#ifndef ENVIRONMENT_HEADER
#define ENVIRONMENT_HEADER

#include "btBulletDynamicsCommon.h"
#include <math.h>
#include <map>
#include <set>
using namespace std;

#include "MeshManager.h"
#include "TextureManager.h"
#include "ShaderManager.h"
#include "RigidBodyComponent.h"
#include "RenderingContext.h"
#include "Renderer.h"

class Action;
class Matrix4f;
class Vector3f;
class Renderer;
class DebugDrawer;
class Camera;
class EnvironmentObject;
class EnvironmentObjectInfo;
class AgentObject;


// TODO: collision shapes sould be cached
class CollisionShapeManager {
private:
	btAlignedObjectArray<btCollisionShape*>	collisionShapes;

public:
	~CollisionShapeManager();
	
	btCollisionShape* getSphereShape(float radius);
	btCollisionShape* getBoxShape(float halfExtentX,
								  float halfExtentY,
								  float halfExtentZ);
	btCollisionShape* getCylinderShape(float halfExtentX,
									   float halfExtentY,
									   float halfExtentZ);
};

class Environment {
private:
	CollisionShapeManager collisionShapeManager;
	btBroadphaseInterface* broadPhase;
	btCollisionDispatcher* dispatcher;
	btConstraintSolver*	solver;
	btDefaultCollisionConfiguration* configuration;
	btDiscreteDynamicsWorld* world;

	AgentObject* agent;
	Renderer renderer;
	int nextObjId;
	set<int> collidedIds;
	map<int, EnvironmentObject*> objectMap; // <obj-id, EnvironmentObject>

	DebugDrawer* debugDrawer;
	MeshManager meshManager;
	TextureManager textureManager;
	ShaderManager shaderManager;
	RenderingContext renderingContext;

	bool initRenderer(int width, int height, const Vector3f& bgColor);
	void prepareAgent();
	void checkCollision();
	void prepareShadow();
	int addObject(btCollisionShape* shape,
				  const Vector3f& pos,
				  float rot,
				  float mass,
				  const Vector3f& relativeCenter,
				  bool detectCollision,
				  const Mesh* mesh,
				  const Vector3f& scale);
	EnvironmentObject* findObject(int id);

	bool prepareDebugDrawer();

public:
	Environment()
		:
		broadPhase(nullptr),
		dispatcher(nullptr),
		solver(nullptr),
		configuration(nullptr),
		world(nullptr),
		agent(nullptr),
		nextObjId(0),
		debugDrawer(nullptr) {
	}

	~Environment() {
	}

	bool init(int width, int height, const Vector3f& bgColor);
	void release();
	void step(const Action& action, int stepNum, bool agentView);
	int addBox(const char* texturePath,
			   const Vector3f& halfExtent,
			   const Vector3f& pos,
			   float rot,
			   float mass,
			   bool detectCollision);
	int addSphere(const char* texturePath,
				  float radius,
				  const Vector3f& pos,
				  float rot,
				  float mass,
				  bool detectCollision);
	int addModel(const char* path,
				 const Vector3f& sale,
				 const Vector3f& pos,
				 float rot,
				 float mass,
				 bool detectCollision);
	void removeObject(int id);
	void locateAgent(const Vector3f& pos,
					 float rot);
	void setLightDir(const Vector3f& dir);
	bool getObjectInfo(int id, EnvironmentObjectInfo& info) const;
	bool getAgentInfo(EnvironmentObjectInfo& info) const;

	const void* getFrameBuffer() const;
	int getFrameBufferWidth() const;
	int getFrameBufferHeight() const;
	int getFrameBufferSize() const;
	void setRenderCamera(const Matrix4f& mat);

	const set<int>& getCollidedIds() const { return collidedIds; }
	void updateCameraToAgentView();
};

#endif
