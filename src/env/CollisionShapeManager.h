// -*- C++ -*-
#ifndef COLLISIONSHAPEMANAGER_HEADER
#define COLLISIONSHAPEMANAGER_HEADER

#include "btBulletDynamicsCommon.h"
#include <map>
#include <string>
using namespace std;

class Vector3f;
class CollisionMeshData;


class CollisionShapeManager {
private:
    map<size_t, btCollisionShape*> collisionShapeMap; // <hash, btCollisionShape>
    
    size_t getHash(const string& str, float v0, float v1=0, float v2=0);
    btCollisionShape* getShape(size_t hash);
    void addShape(size_t hash, btCollisionShape* shape);

public:
    ~CollisionShapeManager();
    
    btCollisionShape* getSphereShape(float radius);
    btCollisionShape* getBoxShape(float halfExtentX,
                                  float halfExtentY,
                                  float halfExtentZ);
    btCollisionShape* getModelShape(const string& path,
                                    const CollisionMeshData& collisionMeshData,
                                    const Vector3f& scale);
    btCollisionShape* getCompoundModelShapeFromFile(const string& path,
                                                    const CollisionMeshData& collisionMeshData,
                                                    const Vector3f& scale);
};

#endif
