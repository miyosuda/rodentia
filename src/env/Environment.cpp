#include "Environment.h"
#include <stdio.h>
#include <functional>

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
#include "CollisionMeshData.h"


bool Environment::init() {
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

    // Linux environment requres GLContext with at least width=1, height=1 size pbuffer.
    bool ret = glContext.init(1, 1);
    return ret;
}

int Environment::addCameraView(int width, int height, const Vector3f& bgColor,
                               float nearClip, float farClip, float focalLength,
                               int shadowBufferWidth) {
    CameraView* cameraView = new CameraView();
    bool ret = cameraView->init(width, height, bgColor, nearClip, farClip, focalLength,
                                shadowBufferWidth);
    if( !ret ) {
        delete cameraView;
        return -1;
    }

    cameraViews.push_back(cameraView);
    return (int)(cameraViews.size()) - 1;
}

int Environment::addAgent(float radius,
                          const Vector3f& pos,
                          float rotY,
                          float mass,
                          bool detectCollision,
                          const Vector3f& color) {
    int objectId = nextObjId;
    nextObjId += 1;

    Texture* texture = textureManager.getColorTexture(color.x, color.y, color.z);
    Shader* shader = shaderManager.getDiffuseShader();
    Shader* shadowDepthShader = shaderManager.getShadowDepthShader();
    Material* material = new Material(texture, shader, shadowDepthShader);
    Mesh* mesh = meshManager.getSphereMesh(material);
    
    btCollisionShape* shape = collisionShapeManager.getSphereShape(radius);

    Vector3f scale(radius, radius, radius);
    AgentObject* agentObj = new AgentObject(mass,
                                            pos,
                                            rotY,
                                            shape,
                                            world,
                                            objectId,
                                            !detectCollision,
                                            mesh,
                                            scale);
    objectMap[objectId] = agentObj;
    return objectId;
}

void Environment::release() {
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

void Environment::checkCollision(CollisionResult& collisionResult) {
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
            const EnvironmentObject* envObj0 = (EnvironmentObject*)obj0->getUserPointer();
            const EnvironmentObject* envObj1 = (EnvironmentObject*)obj1->getUserPointer();

            if( envObj0->isAgent() ) {
                int agentId = envObj0->getObjectId();
                
                const EnvironmentObject* otherObj = envObj1;
                if( !otherObj->ignoresCollision() ) {
                    collisionResult.addCollisionId(agentId, otherObj->getObjectId());

                    if( otherObj->isAgent() ) {
                        // Add inverted agent-agent collision id pair too.
                        collisionResult.addCollisionId(otherObj->getObjectId(), agentId);
                    }
                }
            } else if( envObj1->isAgent() ) {
                int agentId = envObj1->getObjectId();
                
                const EnvironmentObject* otherObj = envObj0;
                if( !otherObj->ignoresCollision() ) {
                    collisionResult.addCollisionId(agentId, otherObj->getObjectId());

                    if( otherObj->isAgent() ) {
                        // Add inverted agent-agent collision id pair too.
                        collisionResult.addCollisionId(otherObj->getObjectId(), agentId);
                    }
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

void Environment::control(int id, const Action& action) {
    if(!world) {
        return;
    }

    EnvironmentObject* envObj = findObject(id);

    if( envObj != nullptr && envObj->isAgent() ) {
        AgentObject* agentObj = (AgentObject*)envObj;
        agentObj->control(action);
    }
}

void Environment::applyImpulse(int id, const Vector3f& impulse) {
    if(!world) {
        return;
    }

    EnvironmentObject* envObj = findObject(id);
    
    if( envObj != nullptr && !envObj->isAgent() ) {
        StageObject* stageObj = (StageObject*)envObj;
        stageObj->applyImpulse(impulse);
    }
}

void Environment::step(CollisionResult& collisionResult) {
    if(!world) {
        return;
    }
    
    const float deltaTime = 1.0f / 60.0f;
    
    world->stepSimulation(deltaTime);

    // Collision check
    checkCollision(collisionResult);

    // Set stage bounding box to rendering context. (currently not used)
    // (LSPSMにbounding boxを設定する予定だが未使用)
    prepareShadow();
}

void Environment::render(int cameraId,
                         const Vector3f& pos,
                         const Quat4f& rot,
                         const set<int> ignoreIds) {

    // Set light direction, ambient color and shadow color rate to the shader
    // TODO: 本来はrender()毎ではなく、step()時に1回だけで良いはず.
    Shader* shader = shaderManager.getDiffuseShader();
    shader->use();
    shader->prepare(renderingContext);

    // Matの計算
    Matrix4f mat;
    mat.set(rot);
    Vector4f pos_(pos.x, pos.y, pos.z, 1.0f);
    mat.setColumn(3, pos_);
    
    // Start shadow rendering path
    // Make depth frame buffer as current
    if( cameraId < 0 || cameraId >= (int)cameraViews.size() ) {
        // TODO: レンダリング失敗の時の対応
        printf("Invaid camera id: %d\n", cameraId);
        return;
    }

    CameraView* cameraView = cameraViews[cameraId];
    cameraView->setCameraMat(mat);

    // Set camera mat to rendering context and and calculate shadow matrix
    // with camera and light direction in LSPSM.
    renderingContext.setCamera(cameraView->getCameraMat(),
                               cameraView->getCameraInvMat(),
                               cameraView->getCameraProjectionMat());

    cameraView->prepareShadowDepthRendering();
    renderingContext.setPath(RenderingContext::SHADOW);

    // Draw objects
    for(auto itr=objectMap.begin(); itr!=objectMap.end(); ++itr) {
        EnvironmentObject* object = itr->second;
        
        if( ignoreIds.find(object->getObjectId()) != ignoreIds.end() ) {
            // Skip drawing
            continue;
        }
        
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
        
        if( ignoreIds.find(object->getObjectId()) != ignoreIds.end() ) {
            // Skip drawing
            continue;
        }
        
        // Draw with normal shader
        object->draw(renderingContext);
    }

    // Read pixels to framebuffer
    cameraView->finishRendering();
}

int Environment::addBox(const char* texturePath,
                        const Vector3f& color,
                        const Vector3f& halfExtent,
                        const Vector3f& pos,
                        const Quat4f& rot,
                        float mass,
                        bool detectCollision,
                        bool visible) {
    btCollisionShape* shape = collisionShapeManager.getBoxShape(halfExtent.x,
                                                                halfExtent.y,
                                                                halfExtent.z);
    Mesh* mesh = nullptr;
    
    if( visible ) {
        Texture* texture = nullptr;
        const string texturePathStr(texturePath);
        if( texturePathStr != "" ) {
            texture = textureManager.loadTexture(texturePath);
            if( texture == nullptr ) {
                texture = textureManager.getColorTexture(1.0f, 1.0f, 1.0f);
            }
        } else {
            texture = textureManager.getColorTexture(color.x, color.y, color.z);
        }
        
        Shader* shader = shaderManager.getDiffuseShader();
        Shader* shadowDepthShader = shaderManager.getShadowDepthShader();
        Material* material = new Material(texture, shader, shadowDepthShader);
        mesh = meshManager.getBoxMesh(material, halfExtent);
    }
    
    Vector3f scale(halfExtent.x, halfExtent.y, halfExtent.z);

    return addObject(shape, pos, rot, mass, Vector3f(0.0f, 0.0f, 0.0f),
                     detectCollision, mesh, scale);
}

int Environment::addSphere(const char* texturePath,
                           const Vector3f& color,
                           float radius,
                           const Vector3f& pos,
                           const Quat4f& rot,
                           float mass,
                           bool detectCollision,
                           bool visible) {
    btCollisionShape* shape = collisionShapeManager.getSphereShape(radius);
    Mesh* mesh = nullptr;
    
    if( visible ) {
        Texture* texture = nullptr;
        const string texturePathStr(texturePath);
        if( texturePathStr != "" ) {
            texture = textureManager.loadTexture(texturePath);
            if( texture == nullptr ) {
                texture = textureManager.getColorTexture(1.0f, 1.0f, 1.0f);
            }
        } else {
            texture = textureManager.getColorTexture(color.x, color.y, color.z);
        }
        
        Shader* shader = shaderManager.getDiffuseShader();
        Shader* shadowDepthShader = shaderManager.getShadowDepthShader();
        Material* material = new Material(texture, shader, shadowDepthShader);
        mesh = meshManager.getSphereMesh(material);
    }
    
    Vector3f scale(radius, radius, radius);
    
    return addObject(shape, pos, rot, mass, Vector3f(0.0f, 0.0f, 0.0f),
                     detectCollision, mesh, scale);
}

int Environment::addModel(const char* path,
                          const Vector3f& color,
                          const Vector3f& scale,
                          const Vector3f& pos,
                          const Quat4f& rot,
                          float mass,
                          bool detectCollision,
                          bool useMeshCollision,
                          bool useCollisionFile,
                          bool visible) {

    Mesh* mesh = nullptr;
    
    // Load mesh from .obj data file
    if( color.x < 0 || color.y < 0 || color.z < 0 ) {
        // When replacing texture was not specified
        mesh = meshManager.getModelMesh(path, textureManager, nullptr, shaderManager);
    } else {
        // When replacing texture was specified
        Texture* texture = textureManager.getColorTexture(color.x, color.y, color.z);
        Shader* shader = shaderManager.getDiffuseShader();
        Shader* shadowDepthShader = shaderManager.getShadowDepthShader();
        Material* replacingMaterial = new Material(texture, shader, shadowDepthShader);
        mesh = meshManager.getModelMesh(path, textureManager, replacingMaterial, shaderManager);
    }
    
    if( mesh == nullptr ) {
        return -1;
    }

    const CollisionMeshData* collisionMeshData = meshManager.getCollisionMeshData(path);
    if( collisionMeshData == nullptr ) {
        return -1;
    }
    
    if(!visible) {
        // TDOO: invisibleの場合でもmeshをロードしているので、無駄があるので、collisionを
        // meshとは別にロードして管理するようにする.
        // (現在先にmeshをロードしないとmeshManager.getCollisionMeshData()で取れない様に
        //  なっているので変更する必要がある)
        mesh = nullptr;
    }

    const BoundingBox& boundingBox = collisionMeshData->getBoundingBox();
    
    Vector3f relativeCenter;
    boundingBox.getCenter(relativeCenter);

    // This relative center offset is used for rigidbody
    relativeCenter.x *= scale.x;
    relativeCenter.y *= scale.y;
    relativeCenter.z *= scale.z;

    btCollisionShape* shape;

    if(useMeshCollision) {
        // Get collision with mesh shape
        shape = collisionShapeManager.getModelShape(path, *collisionMeshData, scale);
    } else if(useCollisionFile) {
        // Get compound collision from collision definition file.
        shape = collisionShapeManager.getCompoundModelShapeFromFile(path,
                                                                    *collisionMeshData,
                                                                    scale);
    } else {
        // Get box collision with the bounding box
        Vector3f halfExtent;
        boundingBox.getHalfExtent(halfExtent);
        
        // We need to apply scale to collision shape in advance.
        halfExtent.x *= scale.x;
        halfExtent.y *= scale.y;
        halfExtent.z *= scale.z;
        shape = collisionShapeManager.getBoxShape(halfExtent.x,
                                                  halfExtent.y,
                                                  halfExtent.z);
    }
    
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
    int objectId = nextObjId;
    nextObjId += 1;

    EnvironmentObject* object = new StageObject(pos,
                                                rot,
                                                mass,
                                                relativeCenter,
                                                shape,
                                                world,
                                                objectId,
                                                !detectCollision,
                                                mesh,
                                                scale);

    objectMap[objectId] = object;
    return objectId;
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

void Environment::locateAgent(int id, const Vector3f& pos, float rotY) {
    locateObject(id, pos, Quat4f(0.0f, sin(rotY * 0.5f), 0.0f, cos(rotY * 0.5f)));
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

const void* Environment::getFrameBuffer(int cameraId) const {
    if( cameraId < 0 || cameraId >= (int)cameraViews.size() ) {
        printf("Invalid camera id: camera_id=%d\n", cameraId);
        return nullptr;
    }
    
    const CameraView* cameraView = cameraViews[cameraId];
    return cameraView->getBuffer();
}

int Environment::getFrameBufferWidth(int cameraId) const {
    if( cameraId < 0 || cameraId >= (int)cameraViews.size() ) {
        printf("Invalid camera id: camera_id=%d\n", cameraId);
        return -1;
    }
    
    const CameraView* cameraView = cameraViews[cameraId];
    return cameraView->getFrameBufferWidth();
}

int Environment::getFrameBufferHeight(int cameraId) const {
    if( cameraId < 0 || cameraId >= (int)cameraViews.size() ) {
        printf("Invalid camera id: camera_id=%d\n", cameraId);
        return -1;
    }
    
    const CameraView* cameraView = cameraViews[cameraId];
    return cameraView->getFrameBufferHeight();
}

int Environment::getFrameBufferSize(int cameraId) const {
    if( cameraId < 0 || cameraId >= (int)cameraViews.size() ) {
        printf("Invalid camera id: camera_id=%d\n", cameraId);
        return -1;
    }
    
    const CameraView* cameraView = cameraViews[cameraId];
    return cameraView->getFrameBufferSize();
}
