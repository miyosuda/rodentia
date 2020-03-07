#include "Environment.h"
#include <stdio.h>

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
#include "CameraView.h"


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

bool Environment::init(int width, int height) {
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

	world->setGravity(btVector3(0, -10, 0));

	// Add agent object
	prepareAgent();

    // TODO: ここのwidth, height引数消せるかどうか調査
    bool ret = glContext.init(width, height);
    return ret;
}

int Environment::addCameraView(int width, int height, const Vector3f& bgColor) {
    /*
	bool ret = initRenderer(width, height, bgColor);
	if( !ret ) {
		return -1;
	}
    */

    CameraView* cameraView = new CameraView();
    bool ret = cameraView->init(width, height, bgColor);
    if( !ret ) {
        delete cameraView;
        return -1;
    }

    cameraViews.push_back(cameraView);
    return (int)(cameraViews.size()) - 1;
}

void Environment::prepareAgent() {
	btCollisionShape* shape = collisionShapeManager.getSphereShape(1.0f);
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
	
	meshManager.release();
	textureManager.release();
	shaderManager.release();

	delete world;
	delete solver;
	delete broadPhase;
	delete dispatcher;
	delete configuration;

	for (auto itr=cameraViews.begin(); itr!=cameraViews.end(); ++itr) {
		CameraView* cameraView = *itr;
        cameraView->release();
		delete cameraView;
	}
    cameraViews.clear();
    glContext.release();
    
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

void Environment::step(const Action& action, int stepNum) {
    if(!world) {
        return;
    }
    
	const float deltaTime = 1.0f/60.0f;
	
    // Process rigid body simulation
    collidedIds.clear();

    for(int i=0; i<stepNum; ++i) {
        if( agent != nullptr ) {
            agent->control(action);
        }

        world->stepSimulation(deltaTime);

        // Collision check
        // (collidedIdsに値をセットする)
        checkCollision();
    }

    // Update agent view camera
    // TODO: 下に持っていけるか.
    // Agentからmatを取ってきて、RenderingContextにmatを設定する.
    // -> cameraにmatをセット
    // -> cameraのmatを利用して LSPSMのshadow matrixを更新する.
    //updateCameraToAgentView();

    // Set stage bounding box to rendering context. (currently not used)
    // (LSPSMにbounding boxを設定する予定だが未使用)
    prepareShadow();
		
    // Set light direction, ambient color and shadow color rate to the shader
    Shader* shader = shaderManager.getDiffuseShader();
    shader->prepare(renderingContext);

    // ここまでは繰り返す必要なし

    // ここからカメラ単位で繰り返し

    // TODO: ここに renderingContext.setCameraMat(mat) を持ってくる?

    /*
    // Start shadow rendering path
    // Make depth frame buffer as current
    renderer.prepareShadowDepthRendering();
    renderingContext.setPath(RenderingContext::SHADOW);

    // Draw objects
    for(auto itr=objectMap.begin(); itr!=objectMap.end(); ++itr) {
        EnvironmentObject* object = itr->second;
        // Draw with shadow depth shader
        object->draw(renderingContext);
    }

    // Start normal rendering path
    // Make normal frame buffer as current
    renderer.prepareRendering();
    renderingContext.setPath(RenderingContext::NORMAL);
		
    // Draw objects
    for(auto itr=objectMap.begin(); itr!=objectMap.end(); ++itr) {
        EnvironmentObject* object = itr->second;
        // Draw with normal shader
        object->draw(renderingContext);
    }

    // Read pixels to framebuffer
    renderer.finishRendering();

    */
}

void Environment::render(int cameraId,
                         const Vector3f& pos,
                         const Quat4f& rot) {

    // Matの計算
    Matrix4f mat;
    mat.set(rot);
    Vector4f pos_(pos.x, pos.y, pos.z, 1.0f);
    mat.setColumn(4, pos_);
    
    setRenderCamera(mat);
    
    // Start shadow rendering path
    // Make depth frame buffer as current
    if( cameraId < 0 || cameraId >= cameraViews.size() ) {
        // TODO: レンダリング失敗の時の対応
        return;
    }
    
    CameraView* cameraView = cameraViews[cameraId];
    
    cameraView->prepareShadowDepthRendering();
    renderingContext.setPath(RenderingContext::SHADOW);

    // Draw objects
    for(auto itr=objectMap.begin(); itr!=objectMap.end(); ++itr) {
        EnvironmentObject* object = itr->second;
        // Draw with shadow depth shader
        object->draw(renderingContext);
    }

    // Start normal rendering path
    // Make normal frame buffer as current
    cameraView->prepareRendering();
    renderingContext.setPath(RenderingContext::NORMAL);
    
    // Draw objects
    for(auto itr=objectMap.begin(); itr!=objectMap.end(); ++itr) {
        EnvironmentObject* object = itr->second;
        // Draw with normal shader
        object->draw(renderingContext);
    }

    // Read pixels to framebuffer
    cameraView->finishRendering();
}

int Environment::addBox(const char* texturePath,
						const Vector3f& halfExtent,
						const Vector3f& pos,
                        const Quat4f& rot,
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
	Mesh* mesh = meshManager.getBoxMesh(material, halfExtent);
	Vector3f scale(halfExtent.x, halfExtent.y, halfExtent.z);

	return addObject(shape, pos, rot, mass, Vector3f(0.0f, 0.0f, 0.0f),
					 detectCollision, mesh, scale);
}

int Environment::addSphere(const char* texturePath,
						   float radius,
						   const Vector3f& pos,
                           const Quat4f& rot,
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
	Mesh* mesh = meshManager.getSphereMesh(material);
	Vector3f scale(radius, radius, radius);
	
	return addObject(shape, pos, rot, mass, Vector3f(0.0f, 0.0f, 0.0f),
					 detectCollision, mesh, scale);
}

int Environment::addModel(const char* path,
						  const Vector3f& scale,
						  const Vector3f& pos,
                          const Quat4f& rot,
						  float mass,
						  bool detectCollision) {

	// Load mesh from .obj data file
	Mesh* mesh = meshManager.getModelMesh(path, textureManager, shaderManager);
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
                           const Quat4f& rot,
						   float mass,
						   const Vector3f& relativeCenter,
						   bool detectCollision,
						   Mesh* mesh,
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

void Environment::locateObject(int id, const Vector3f& pos, const Quat4f& rot) {
	auto itr = objectMap.find(id);
	if( itr != objectMap.end() ) {
		EnvironmentObject* object = objectMap[id];
		object->locate(pos, rot);
	}
}

void Environment::locateAgent(const Vector3f& pos, float angle) {
	if( agent != nullptr ) {
        // TODO: 要確認
		agent->locate(pos, Quat4f(0.0f, sin(angle * 0.5f), 0.0f, cos(angle * 0.5f)));
	}
}

void Environment::setLight(const Vector3f& lightDir,
						   const Vector3f& lightColor,
						   const Vector3f& ambientColor,
						   float shadowColorRate) {
	renderingContext.setLight(lightDir,
							  lightColor,
							  ambientColor,
							  shadowColorRate);
}

bool Environment::getObjectInfo(int id, EnvironmentObjectInfo& info) const {
	auto itr = objectMap.find(id);
	if( itr != objectMap.end() ) {
		const EnvironmentObject* object = itr->second;
		object->getInfo(info);
		return true;
	} else {
		return false;
	}
}

bool Environment::getAgentInfo(EnvironmentObjectInfo& info) const {
	if( agent != nullptr ) {
		agent->getInfo(info);
		return true;
	} else {
		return false;
	}
}

void Environment::replaceObjectTextures(int id, const vector<string>& texturePathes) {
	auto itr = objectMap.find(id);
	if( itr == objectMap.end() ) {
		printf("Failed to find object: id=%d\n", id);
		return;
	}

	EnvironmentObject* object = objectMap[id];
	
	vector<Material*> materials;

	for(unsigned int i=0; i<texturePathes.size(); ++i) {
		Texture* texture = textureManager.loadTexture(texturePathes[i].c_str());
		if( texture != nullptr ) {
			Shader* shader = shaderManager.getDiffuseShader();
			Shader* shadowDepthShader = shaderManager.getShadowDepthShader();
			Material* material = new Material(texture, shader, shadowDepthShader);
			materials.push_back(material);
		}
	}

	object->replaceMaterials(materials);
}

/*
bool Environment::initRenderer(int width, int height, const Vector3f& bgColor) {
    // TODO: RenderTargetの設定
	bool ret = renderer.init(width, height, bgColor);
	if(!ret) {
		return false;
	}

    // TODO: カメラの設定
	float ratio = width / (float) height;
	renderingContext.initCamera(ratio);

	return true;
}
*/

const void* Environment::getFrameBuffer(int cameraId) const {
    if( cameraId < 0 || cameraId >= cameraViews.size() ) {
        // TODO: 取得失敗の時の対応
        return nullptr;
    }
    
    const CameraView* cameraView = cameraViews[cameraId];
    return cameraView->getBuffer();
}

int Environment::getFrameBufferWidth(int cameraId) const {
    if( cameraId < 0 || cameraId >= cameraViews.size() ) {
        // TODO: 取得失敗の時の対応
        return -1;
    }
    
    const CameraView* cameraView = cameraViews[cameraId];
    return cameraView->getFrameBufferWidth();
}

int Environment::getFrameBufferHeight(int cameraId) const {
    if( cameraId < 0 || cameraId >= cameraViews.size() ) {
        // TODO: 取得失敗の時の対応
        return -1;
    }
    
    const CameraView* cameraView = cameraViews[cameraId];
    return cameraView->getFrameBufferHeight();
}

int Environment::getFrameBufferSize(int cameraId) const {
    if( cameraId < 0 || cameraId >= cameraViews.size() ) {
        // TODO: 取得失敗の時の対応
        return -1;
    }
    
    const CameraView* cameraView = cameraViews[cameraId];
    return cameraView->getFrameBufferSize();
}


void Environment::setRenderCamera(const Matrix4f& mat) {
    // Set camera mat to rendering context and and calculate shadow matrix
    // with camera and light direction in LSPSM.
    renderingContext.setCameraMat(mat);
}

/*
void Environment::updateCameraToAgentView() {
	if( agent != nullptr ) {
		Matrix4f mat;
		agent->getMat(mat);
		setRenderCamera(mat);
	}
}
*/
