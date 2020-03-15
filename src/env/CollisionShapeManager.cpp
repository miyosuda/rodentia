#include "CollisionShapeManager.h"
#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>

#include "Vector3f.h"
#include "CollisionMeshData.h"


CollisionShapeManager::~CollisionShapeManager() {
    // Delete collision shapes
    for(auto itr=collisionShapeMap.begin(); itr!=collisionShapeMap.end(); ++itr) {
        auto shape = itr->second;
        delete shape;
    }
}

size_t CollisionShapeManager::getHash(const string& str, float v0, float v1, float v2) {
    size_t hashStr = std::hash<string>()(str);
    size_t hashV0 = std::hash<float>()(v0);
    size_t hashV1 = std::hash<float>()(v1);
    size_t hashV2 = std::hash<float>()(v2);
    return hashStr + 1009 * hashV0 + 2131 * hashV1 + 3571 * hashV2;
}

btCollisionShape* CollisionShapeManager::getShape(size_t hash) {
    auto itr = collisionShapeMap.find(hash);
    if( itr != collisionShapeMap.end() ) {
        return collisionShapeMap[hash];
    } else {
        return nullptr;
    }
}

void CollisionShapeManager::addShape(size_t hash, btCollisionShape* shape) {
    collisionShapeMap[hash] = shape;
}

btCollisionShape* CollisionShapeManager::getSphereShape(float radius) {
    size_t hash = getHash("@sphere", radius);
    btCollisionShape* shape = getShape(hash);
    if( shape != nullptr ) {
        return shape;
    }
    
    shape = new btSphereShape(radius);
    addShape(hash, shape);
    return shape;
}

btCollisionShape* CollisionShapeManager::getBoxShape(float halfExtentX,
                                                     float halfExtentY,
                                                     float halfExtentZ) {
    size_t hash = getHash("@box", halfExtentX, halfExtentY, halfExtentZ);
    btCollisionShape* shape = getShape(hash);
    if( shape != nullptr ) {
        return shape;
    }
    
    shape = new btBoxShape(btVector3(halfExtentX,
                                     halfExtentY,
                                     halfExtentZ));
    addShape(hash, shape);
    return shape;
}

btCollisionShape* CollisionShapeManager::getModelShape(
    const string& path,
    const CollisionMeshData& collisionMeshData,
    const Vector3f& scale) {

    size_t hash = getHash(path, scale.x, scale.y, scale.z);
    btCollisionShape* shape = getShape(hash);
    if( shape != nullptr ) {
        return shape;
    }
    
    const BoundingBox& boundingBox = collisionMeshData.getBoundingBox();
    
    Vector3f center;
    boundingBox.getCenter(center);

    // This  center offset is used for rigidbody
    center.x *= scale.x;
    center.y *= scale.y;
    center.z *= scale.z;

    btTriangleMesh* trimesh = new btTriangleMesh();
        
    auto triangles = collisionMeshData.getTriangles();
    for(auto itr=triangles.begin(); itr!=triangles.end(); ++itr) {
        trimesh->addTriangle(btVector3(itr->x0 * scale.x - center.x,
                                       itr->y0 * scale.y - center.y,
                                       itr->z0 * scale.z - center.z),
                             btVector3(itr->x1 * scale.x - center.x,
                                       itr->y1 * scale.y - center.y,
                                       itr->z1 * scale.z - center.z),
                             btVector3(itr->x2 * scale.x - center.x,
                                       itr->y2 * scale.y - center.y,
                                       itr->z2 * scale.z - center.z));
    }

    const bool useQuantizedBvhTree = true;
    shape = new btBvhTriangleMeshShape(trimesh, useQuantizedBvhTree);
    addShape(hash, shape);
    return shape;
}

static istream& safeGetline(istream &is, string &t) {
    t.clear();
  
    istream::sentry se(is, true);
    streambuf *sb = is.rdbuf();
  
    if (se) {
        for (;;) {
            int c = sb->sbumpc();
            switch (c) {
            case '\n':
                return is;
            case '\r':
                if (sb->sgetc() == '\n') {
                    sb->sbumpc();
                }
                return is;
            case EOF:
                if (t.empty()) {
                    is.setstate(ios::eofbit);
                }
                return is;
            default:
                t += static_cast<char>(c);
            }
        }
    }

    return is;
}

const char DELIMITER = ' ';

static void tokenize(string str, vector<string> &token_v) {
    size_t start = str.find_first_not_of(DELIMITER);
    size_t end = start;

    while(start != string::npos) {
        end = str.find(DELIMITER, start);
        token_v.push_back(str.substr(start, end-start));
        start = str.find_first_not_of(DELIMITER, end);
    }
}


btCollisionShape* CollisionShapeManager::getCompoundModelShapeFromFile(
    const string& path,
    const CollisionMeshData& collisionMeshData,
    const Vector3f& scale) {

   size_t pos = path.find_last_of(".");
    if( pos == string::npos ) {
        return nullptr;
    }

    const string colFilePath = path.substr(0, pos+1) + "col";

    ifstream is(colFilePath.c_str());
    if(!is) {
        return nullptr;
    }

    size_t hash = getHash(colFilePath, scale.x, scale.y, scale.z);
    btCollisionShape* shape = getShape(hash);
    if( shape != nullptr ) {
        return shape;
    }
    
    btCompoundShape* compoundShape = new btCompoundShape();

    const BoundingBox& boundingBox = collisionMeshData.getBoundingBox();
    
    Vector3f center;
    boundingBox.getCenter(center);

    // This  center offset is used for rigidbody
    center.x *= scale.x;
    center.y *= scale.y;
    center.z *= scale.z;
    
    string linebuf;
    while(is.peek() != -1) {
        safeGetline(is, linebuf);
        
        if( linebuf[0] == 'b' ) {
            vector<string> tokens;
            tokenize(linebuf, tokens);
            if( tokens.size() >= 7 ) {
                float posX = stof(tokens[1]);
                float posY = stof(tokens[2]);
                float posZ = stof(tokens[3]);
                
                float halfExtentX = stof(tokens[4]);
                float halfExtentY = stof(tokens[5]);
                float halfExtentZ = stof(tokens[6]);
                

                btCollisionShape* boxShape = getBoxShape(halfExtentX * scale.x,
                                                         halfExtentY * scale.y,
                                                         halfExtentZ * scale.z);
                
                btTransform transform;
                transform.setIdentity();
                transform.setOrigin(btVector3(posX * scale.x - center.x,
                                              posY * scale.y - center.y,
                                              posZ * scale.z - center.z));
                compoundShape->addChildShape(transform, boxShape);
            }
        }
        
        if(linebuf.empty()) {
            continue;
        }
    }
    
    compoundShape->recalculateLocalAabb();
    addShape(hash, compoundShape);
    return compoundShape;
}
