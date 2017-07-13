// -*- C++ -*-
#ifndef ENVIRONMENT_HEADER
#define ENVIRONMENT_HEADER

#include "btBulletDynamicsCommon.h"
#include <math.h>
#include <map>
#include <vector>
using namespace std;

#include "ShaderManager.h"

class Action {
public:
	int look;   // look left=[+], look right=[-]
	int strafe; // strafe left=[+1], strafe right=[-1]
	int move;   // forward=[+1], backward=[-1]

	Action()
		:
		look(0),
		strafe(0),
		move(0) {
	}

	Action(int look_, int strafe_, int move_)
		:
		look(look_),
		strafe(strafe_),
		move(move_) {
	}

	void set(int look_, int strafe_, int move_) {
		look   = look_;
		strafe = strafe_;
		move   = move_;
	}

	static int getActionSize() {
		return 3;
	}
};

class Matrix4f;

class Model {
private:
	btDynamicsWorld*  world;
	btCollisionShape* shape;
	btRigidBody*	  body;

public:
	Model(btDynamicsWorld* world_, btRigidBody* floorBody);
	~Model();

	void control(const Action& action);
	void getMat(Matrix4f& mat) const;
};

class Renderer;
class DebugDrawer;

class Environment {
	btAlignedObjectArray<btCollisionShape*>	collisionShapes;
	btBroadphaseInterface* broadPhase;
	btCollisionDispatcher* dispatcher;
	btConstraintSolver*	solver;
	btDefaultCollisionConfiguration* configuration;
	btDiscreteDynamicsWorld* world;

	Model* model;
	Renderer* renderer;
	int nextObjId;
	vector<int> collidedIds;
	map<int, btRigidBody*> bodyMap;

	DebugDrawer* debugDrawer;
	ShaderManager shaderManager; //..

	void checkCollision();
	btRigidBody* createBox(float halfExtentX, float halfExtentY, float halfExtentZ,
						   float posX, float posY, float posZ,
						   float rot);

public:
	Environment()
		:
		broadPhase(nullptr),
		dispatcher(nullptr),
		solver(nullptr),
		configuration(nullptr),
		world(nullptr),
		model(nullptr),
		renderer(nullptr),
		nextObjId(0),
		debugDrawer(nullptr) {
	}

	~Environment() {
	}

	void init();
	void release();
	void step(const Action& action, bool updateCamera=false);
	int addBox(float halfExtentX, float halfExtentY, float halfExtentZ,
			   float posX, float posY, float posZ,
			   float rot,
			   bool detectCollision);
	int addSphere(float radius,
				  float posX, float posY, float posZ,
				  float rot,
				  bool detectCollision);
	void removeObj(int id);
	void locateAgent(float posX, float posY, float posZ,
					 float rot);

	bool initRenderer(int width, int height, bool offscreen);
	const void* getFrameBuffer() const;
	int getFrameBufferWidth() const;
	int getFrameBufferHeight() const;
	int getFrameBufferSize() const;
	void setRenderCamera(const Matrix4f& mat);

	const vector<int>& getCollidedIds() const { return collidedIds; }
	void updateCameraToAgentView();
};

#endif
