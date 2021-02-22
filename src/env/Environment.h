// -*- C++ -*-
#ifndef ENVIRONMENT_HEADER
#define ENVIRONMENT_HEADER

#include "btBulletDynamicsCommon.h"
#include <math.h>
#include <map>
#include <set>
#include <vector>
#include <string>
using namespace std;

#include "MeshManager.h"
#include "TextureManager.h"
#include "ShaderManager.h"
#include "RigidBodyComponent.h"
#include "RenderingContext.h"
#include "glinc.h"
#include "GLContext.h"
#include "CollisionShapeManager.h"

class Action;
class Matrix4f;
class Vector3f;
class Camera;
class EnvironmentObject;
class EnvironmentObjectInfo;
class AgentObject;
class CameraView;



class CollisionResult {
private:
    map< int, set<int> > collisionIdMap; // <agentId, collisionids>

public:
    CollisionResult() {
    }

    void addCollisionId(int agentId, int targetId) {
        collisionIdMap[agentId].insert(targetId);
    }

    void getAgentIds(vector<int>& agentIds) const {
        for(auto itr=collisionIdMap.begin(); itr!=collisionIdMap.end(); ++itr) {
            int id = itr->first;            
            agentIds.push_back(id);
        }
    }

    void getCollisionIds(int agentId, vector<int>& collisionIds) const {
        auto itr = collisionIdMap.find(agentId);
        if( itr == collisionIdMap.end() ) {
            return;
        }

        const set<int>& collidedIds = itr->second;
        for(auto it=collidedIds.begin(); it!=collidedIds.end(); ++it) {
            int id = *it;
            collisionIds.push_back(id);
        }
    }
};


class Environment {
private:
    CollisionShapeManager collisionShapeManager;
    btBroadphaseInterface* broadPhase;
    btCollisionDispatcher* dispatcher;
    btConstraintSolver* solver;
    btDefaultCollisionConfiguration* configuration;
    btDiscreteDynamicsWorld* world;

    int nextObjId;
    map<int, EnvironmentObject*> objectMap; // <obj-id, EnvironmentObject>

    MeshManager meshManager;
    TextureManager textureManager;
    ShaderManager shaderManager;
    GLContext glContext;
    RenderingContext renderingContext;
    vector<CameraView*> cameraViews;

    void checkCollision(CollisionResult& collisionResult);
    void prepareShadow();
    int addObject(btCollisionShape* shape,
                  const Vector3f& pos,
                  const Quat4f& rot,
                  float mass,
                  const Vector3f& relativeCenter,
                  bool detectCollision,
                  Mesh* mesh,
                  const Vector3f& scale);
    EnvironmentObject* findObject(int id);

public:
    Environment()
        :
        broadPhase(nullptr),
        dispatcher(nullptr),
        solver(nullptr),
        configuration(nullptr),
        world(nullptr),
        nextObjId(0) {
    }

    ~Environment() {
    }

    bool init();
    int addCameraView(int width, int height, const Vector3f& bgColor,
                      float nearClip, float farClip, float focalLength,
                      int shadowBufferWidth);
    int addAgent(float radius,
                 const Vector3f& pos,
                 float rotY,
                 float mass,
                 bool detectCollision,
                 const Vector3f& color);
    void release();
    void control(int id, const Action& action);
    void applyImpulse(int id, const Vector3f& impulse);
    void step(CollisionResult& collisionResult);
    int addBox(const char* texturePath,
               const Vector3f& color,
               const Vector3f& halfExtent,
               const Vector3f& pos,
               const Quat4f& rot,
               float mass,
               bool detectCollision,
               bool visible);
    int addSphere(const char* texturePath,
                  const Vector3f& color,
                  float radius,
                  const Vector3f& pos,
                  const Quat4f& rot,
                  float mass,
                  bool detectCollision,
                  bool visible);
    int addModel(const char* path,
                 const Vector3f& color,
                 const Vector3f& sale,
                 const Vector3f& pos,
                 const Quat4f& rot,
                 float mass,
                 bool detectCollision,
                 bool useMeshCollision,
                 bool useCollisionFile,
                 bool visible);
    void removeObject(int id);
    void locateObject(int id, const Vector3f& pos, const Quat4f& rot);
    void locateAgent(int id, const Vector3f& pos, float rotY);
    void setLight(const Vector3f& lightDir,
                  const Vector3f& lightColor,
                  const Vector3f& ambientColor,
                  float shadowColorRate);
    bool getObjectInfo(int id, EnvironmentObjectInfo& info) const;
    void replaceObjectTextures(int id, const vector<string>& texturePathes);

    void render(int cameraId, const Vector3f& pos, const Quat4f& rot,
                const set<int> ignoreIds);
    const void* getFrameBuffer(int cameraId) const;
    int getFrameBufferWidth(int cameraId) const;
    int getFrameBufferHeight(int cameraId) const;
    int getFrameBufferSize(int cameraId) const;
};

#endif
