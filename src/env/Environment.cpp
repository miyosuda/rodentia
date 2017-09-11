#include "Environment.h"
#include <stdio.h>

#include "DebugDrawer.h"
#include "DrawComponent.h"
#include "Mesh.h"
#include "Vector3f.h"
#include "Camera.h"
#include "Material.h"
#include "Texture.h"
#include "Action.h"
#include "Shader.h"
#include "EnvironmentObject.h"
#include "BoundingBox.h"

static const int ID_IGNORE_COLLISION = -1;
static const int ID_AGENT = -2;


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

bool Environment::init(int width, int height, const Vector3f& bgColor) {
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

	bool ret = initRenderer(width, height, bgColor);
	if( !ret ) {
		return false;
	}

	world->setGravity(btVector3(0, -10, 0));

	// Add agent object
	prepareAgent();

	return true;
}

void Environment::prepareAgent() {
	btCollisionShape* shape = collisionShapeManager.getCylinderShape(0.25f, 1.0f, 0.25f);
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

	meshManager.release();
	textureManager.release();
	shaderManager.release();

	delete world;
	delete solver;
	delete broadPhase;
	delete dispatcher;
	delete configuration;

	renderer.release();
	nextObjId = 0;
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
				if( otherId != ID_AGENT && otherId != ID_IGNORE_COLLISION ) {
					collidedIds.insert(otherId);
				}
			} else if( obj1->getUserIndex() == ID_AGENT ) {
				int otherId = obj0->getUserIndex();
				if( otherId != ID_AGENT && otherId != ID_IGNORE_COLLISION ) {
					collidedIds.insert(otherId);
				}
			}
		}
	}
}

void Environment::prepareShadow() {
	// Calculate bouding box
	BoundingBox stageBoundingBox;
	for(auto itr=objectMap.begin(); itr!=objectMap.end(); ++itr) {
		EnvironmentObject* object = itr->second;
		BoundingBox boundingBox;
		if( object->calcBoundingBox(boundingBox) ) {
			stageBoundingBox.merge(boundingBox);
		}
	}

	if( stageBoundingBox.isInitalized() ) {
		renderingContext.setBoundingBoxForShadow(stageBoundingBox);
	}
}

void Environment::step(const Action& action, int stepNum, bool agentView) {
	const float deltaTime = 1.0f/60.0f;
	
	if(world) {
		// Process rigid body simulation
		collidedIds.clear();

		for(int i=0; i<stepNum; ++i) {
			if( agent != nullptr ) {
				agent->control(action);
			}

			world->stepSimulation(deltaTime);

			// Collision check
			checkCollision();
		}

		// Update agent view camera
		if( agentView ) {
			updateCameraToAgentView();
		}

		prepareShadow();
		
		// Set light direction to shader
		Shader* shader = shaderManager.getDiffuseShader();
		shader->prepare(renderingContext);

		// Start shadow rendering path
		renderer.prepareShadowDepthRendering();
		renderingContext.setPath(RenderingContext::SHADOW);

		// Draw objects
		for(auto itr=objectMap.begin(); itr!=objectMap.end(); ++itr) {
			EnvironmentObject* object = itr->second;
			object->draw(renderingContext);
		}

		// Start normal rendering path
		renderer.prepareRendering();
		renderingContext.setPath(RenderingContext::NORMAL);
		
		// Draw objects
		for(auto itr=objectMap.begin(); itr!=objectMap.end(); ++itr) {
			EnvironmentObject* object = itr->second;
			object->draw(renderingContext);
		}

		// Debug drawing
		if( debugDrawer != nullptr) {
			// TODO: not drawn when drawing object mesh.
			glDisable(GL_DEPTH_TEST);
			debugDrawer->prepare(renderingContext);
			world->debugDrawWorld();
			glEnable(GL_DEPTH_TEST);
		}

		renderer.finishRendering();
	}
}

int Environment::addBox(const char* texturePath,
						const Vector3f& halfExtent,
						const Vector3f& pos,
						float rot,
						float mass,
						bool detectCollision) {
	btCollisionShape* shape = collisionShapeManager.getBoxShape(halfExtent.x,
																halfExtent.y,
																halfExtent.z);
	Texture* texture = nullptr;

	const string texturePathStr(texturePath);
	if( texturePathStr != "" ) {
		texture = textureManager.loadTexture(texturePath);
	}
	if( texture == nullptr ) {
		texture = textureManager.getColorTexture(1.0f, 1.0f, 1.0f);
	}
	Shader* shader = shaderManager.getDiffuseShader();
	Shader* shadowDepthShader = shaderManager.getShadowDepthShader();
	Material* material = new Material(texture, shader, shadowDepthShader);
	const Mesh* mesh = meshManager.getBoxMesh(material, halfExtent);
	Vector3f scale(halfExtent.x, halfExtent.y, halfExtent.z);

	return addObject(shape, pos, rot, mass, Vector3f(0.0f, 0.0f, 0.0f),
					 detectCollision, mesh, scale);
}

int Environment::addSphere(const char* texturePath,
						   float radius,
						   const Vector3f& pos,
						   float rot,
						   float mass,
						   bool detectCollision) {
	btCollisionShape* shape = collisionShapeManager.getSphereShape(radius);
	
	Texture* texture = nullptr;
	if( texturePath != nullptr) {
		texture = textureManager.loadTexture(texturePath);
	}
	if( texture == nullptr ) {
		texture = textureManager.getColorTexture(1.0f, 1.0f, 1.0f);
	}
	Shader* shader = shaderManager.getDiffuseShader();
	Shader* shadowDepthShader = shaderManager.getShadowDepthShader();
	Material* material = new Material(texture, shader, shadowDepthShader);
	const Mesh* mesh = meshManager.getSphereMesh(material);
	Vector3f scale(radius, radius, radius);
	
	return addObject(shape, pos, rot, mass, Vector3f(0.0f, 0.0f, 0.0f),
					 detectCollision, mesh, scale);
}

int Environment::addModel(const char* path,
						  const Vector3f& scale,
						  const Vector3f& pos,
						  float rot,
						  float mass,
						  bool detectCollision) {

	// Load mesh from .obj data file
	const Mesh* mesh = meshManager.getModelMesh(path, textureManager, shaderManager);
	if( mesh == nullptr ) {
		return -1;
	}

	const BoundingBox& boundingBox = mesh->getBoundingBox();
	
	Vector3f relativeCenter;
	Vector3f halfExtent;
	boundingBox.getCenter(relativeCenter);
	boundingBox.getHalfExtent(halfExtent);
	
	halfExtent.x *= scale.x;
	halfExtent.y *= scale.y;
	halfExtent.z *= scale.z;

	// We need to apply scale to collision shape in advance.
	btCollisionShape* shape = collisionShapeManager.getBoxShape(halfExtent.x,
																halfExtent.y,
																halfExtent.z);

	// This relative center offset is used for rigidbody
	relativeCenter.x *= scale.x;
	relativeCenter.y *= scale.y;
	relativeCenter.z *= scale.z;
	
	return addObject(shape, pos, rot, mass, relativeCenter, detectCollision, mesh, scale);
}

int Environment::addObject(btCollisionShape* shape,
						   const Vector3f& pos,
						   float rot,
						   float mass,
						   const Vector3f& relativeCenter,
						   bool detectCollision,
						   const Mesh* mesh,
						   const Vector3f& scale) {
	int id = nextObjId;
	nextObjId += 1;

	int collisionId;

	if( detectCollision ) {
		collisionId = id;
	} else {
		collisionId = ID_IGNORE_COLLISION;
	}

	EnvironmentObject* object = new StageObject(
		pos,
		rot,
		mass,
		relativeCenter,
		shape,
		world,
		collisionId,
		mesh,
		scale);

	objectMap[id] = object;
	return id;
}

EnvironmentObject* Environment::findObject(int id) {
	auto itr = objectMap.find(id);
	if( itr != objectMap.end() ) {
		EnvironmentObject* object = objectMap[id];
		return object;
	} else {
		return nullptr;
	}
}

void Environment::removeObject(int id) {
	auto itr = objectMap.find(id);
	if( itr != objectMap.end() ) {
		EnvironmentObject* object = objectMap[id];
		delete object;
		objectMap.erase(itr);
	}
}

void Environment::locateAgent(const Vector3f& pos, float rot) {
	if( agent != nullptr ) {
		// TODO: Use raycast to find agent Y pos.
		Vector3f adjustedPos(pos);
		adjustedPos.y = 1.0f;
		agent->locate(adjustedPos, rot);
	}
}

void Environment::setLightDir(const Vector3f& dir) {
	renderingContext.setLightDir(dir);
}

bool Environment::prepareDebugDrawer() {
	Shader* lineShader = shaderManager.getLineShader();
	// Set debug drawer
	debugDrawer = new DebugDrawer(lineShader);
	bool ret = debugDrawer->init();
	if(!ret) {
		return false;
	}
	
	world->setDebugDrawer(debugDrawer);
	int debugMode =
		btIDebugDraw::DBG_DrawWireframe |
		btIDebugDraw::DBG_DrawConstraints |
		btIDebugDraw::DBG_DrawConstraintLimits;
	debugDrawer->setDebugMode(debugMode);

	return true;
}

bool Environment::initRenderer(int width, int height, const Vector3f& bgColor) {
	bool ret = renderer.init(width, height, bgColor);
	if(!ret) {
		return false;
	}

	float ratio = width / (float) height;
	renderingContext.initCamera(ratio);

	// Set debug drawer
	/*
	ret = prepareDebugDrawer();
	if(!ret) {
		return false;
	}
	*/

	return true;
}

const void* Environment::getFrameBuffer() const {
	return renderer.getBuffer();
}

int Environment::getFrameBufferWidth() const {
	return renderer.getFrameBufferWidth();
}

int Environment::getFrameBufferHeight() const {
	return renderer.getFrameBufferHeight();
}

int Environment::getFrameBufferSize() const {
	return renderer.getFrameBufferSize();
}

void Environment::setRenderCamera(const Matrix4f& mat) {
	renderingContext.setCameraMat(mat);
}

void Environment::updateCameraToAgentView() {
	if( agent != nullptr ) {
		Matrix4f mat;
		agent->getMat(mat);
		setRenderCamera(mat);
	}
}
