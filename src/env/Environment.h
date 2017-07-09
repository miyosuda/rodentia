// -*- C++ -*-
#ifndef ENVIRONMENT_HEADER
#define ENVIRONMENT_HEADER

#include "btBulletDynamicsCommon.h"
#include <math.h>
#include <map>
#include <vector>
using namespace std;

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

class Model {
private:
	btDynamicsWorld*  world;
	btCollisionShape* shape;
	btRigidBody*	  body;

public:
	Model(btDynamicsWorld* world_);
	~Model();

	void control(const Action& action);
};

class Renderer;
class Matrix4f;

class Environment {
	// [typedef]
	typedef map<int, btRigidBody*> BodyMap; // <id,btRigidBody*>
	//typedef BodyMap::const_iterator ConstBodyMapIterator;
	//typedef BodyMap::iterator       BodyMapIterator;
	
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
	BodyMap bodyMap;

	void checkCollision();

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
		nextObjId(0) {
	}

	~Environment() {
	}

	void init();
	void release();
	void step(const Action& action);
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
};

#endif
