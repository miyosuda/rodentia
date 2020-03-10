// -*- C++ -*-
#ifndef ENVIRONMENTOBJECT_HEADER
#define ENVIRONMENTOBJECT_HEADER

#include <vector>
using namespace std;

#include "btBulletDynamicsCommon.h"
#include "Matrix4f.h"
#include "RigidBodyComponent.h"

class RigidBodyComponent;
class DrawComponent;
class Action;
class Mesh;
class RenderingContext;
class BoundingBox;
class Material;

//---------------------------
// [EnvironmentObjectInfo]
//---------------------------
class EnvironmentObjectInfo {
public:
    Vector3f pos;
    Vector3f velocity;
    Quat4f rot;
    
public: 
    void set(const Matrix4f& mat, const Vector3f& velocity_);
};


//---------------------------
//   [EnvironmentObject]
//---------------------------
class EnvironmentObject {
protected:
    RigidBodyComponent* rigidBodyComponent;
    DrawComponent* drawComponent;

public:
    EnvironmentObject();
    virtual ~EnvironmentObject();
    int getCollisionId() const;
    void getMat(Matrix4f& mat) const;
    void draw(RenderingContext& context) const;
    bool calcBoundingBox(BoundingBox& boundingBox);
    void getInfo(EnvironmentObjectInfo& info) const;
    void locate(const Vector3f& pos, const Quat4f& rot);
    void replaceMaterials(const vector<Material*>& materials);
};


//---------------------------
//      [StageObject]
//---------------------------
class StageObject : public EnvironmentObject {
public:
    StageObject(const Vector3f& pos,
                const Quat4f& rot,
                float mass,
                const Vector3f& relativeCenter,
                btCollisionShape* shape,
                btDynamicsWorld* world,
                int collisionId,
                Mesh* mesh,
                const Vector3f& scale);
};

//---------------------------
//      [AgentObject]
//---------------------------
class AgentObject : public EnvironmentObject {
public: 
    AgentObject(btCollisionShape* shape,
                btDynamicsWorld* world,
                int collisionId);
    void control(const Action& action);
};

#endif
