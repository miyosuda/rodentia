#include "ObjImporter.h"
#include "MeshData.h"
#include "MeshFaceData.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include <cstdint> // for int64_t

struct ObjVertex {
    float vx;
    float vy;
    float vz;
    float nx;
    float ny;
    float nz;
    float u;
    float v;

    ObjVertex(float vx_,
              float vy_,
              float vz_,
              float nx_,
              float ny_,
              float nz_,
              float u_,
              float v_) {
        vx = vx_;
        vy = vy_;
        vz = vz_;
        nx = nx_;
        ny = ny_;
        nz = nz_;
        u = u_;
        v = v_;
    }
};

class ObjMeshFace {
private:
    std::vector<ObjVertex> vertices;
    std::map<int64_t, int> indexMap;
    std::vector<unsigned short> indices;
    std::string texturePath;

    int64_t getIndexHash(int vertexIndex, int normalIndex, int texcoordIndex) {
        return
            (int64_t)(vertexIndex+1) +
            (int64_t)(normalIndex+1) * 0x1000000L +
            (int64_t)(texcoordIndex+1) * 0x1000000000000L;
    }

public:
    void addVertex(int vertexIndex,
                   int normalIndex,
                   int texcoordIndex,
                   float vx,
                   float vy,
                   float vz,
                   float nx,
                   float ny,
                   float nz,
                   float u,
                   float v) {
        int64_t indexHash = getIndexHash(vertexIndex,
                                         normalIndex,
                                         texcoordIndex);
        auto itr = indexMap.find(indexHash);

        int index;
        if( itr != indexMap.end() ) {
            index = itr->second;
        } else {
            ObjVertex objVertex(vx, vy, vz, nx, ny, nz, u, v);
            vertices.push_back(objVertex);
            index = static_cast<int>(vertices.size() - 1);
        }
        indices.push_back(static_cast<unsigned>(index));
    }

    void setTexturePath(const std::string& texturePath_) {
        texturePath = texturePath_;
    }

    MeshFaceData* toMeshFaceData() const {
        float* vertices_ = new float[vertices.size() * 8];
        unsigned short* indices_ = new unsigned short[indices.size()];

        int verticesSize = vertices.size();
        int indicesSize = indices.size();
        for(int i=0; i<verticesSize; ++i) {
            const ObjVertex& v = vertices[i];
            vertices_[8*i+0] = v.vx;
            vertices_[8*i+1] = v.vy;
            vertices_[8*i+2] = v.vz;
            vertices_[8*i+3] = v.nx;
            vertices_[8*i+4] = v.ny;
            vertices_[8*i+5] = v.nz;
            vertices_[8*i+6] = v.u;
            vertices_[8*i+7] = v.v;
        }

        for(int i=0; i<indicesSize; ++i) {
            indices_[i] = indices[i];
        }

        MeshFaceData* meshFaceData = new MeshFaceData();
        bool ret = meshFaceData->init(vertices_,
                                      verticesSize * 8,
                                      indices_,
                                      indicesSize);
        
        delete [] vertices_;
        delete [] indices_;
        
        if( !ret ) {
            return nullptr;
        }
        
        return meshFaceData;
    }

    const string& getTexturePath() const {
        return texturePath;
    }
};

/**
 * <!--  loadObjMesh():  -->
 */
static MeshData* loadObjMeshData(const tinyobj::attrib_t& attrib,
                                 const std::vector<tinyobj::shape_t>& shapes,
                                 const std::vector<tinyobj::material_t>& materials,
                                 const std::string& dirPath) {

    std::vector<ObjMeshFace> meshFaces;

    int meshFaceSize = materials.size();
    if( meshFaceSize == 0 ) {
        meshFaceSize = 1;
    }

    meshFaces.resize(meshFaceSize);

    if( materials.size() == 0 ) {
        meshFaces[0].setTexturePath("");
    } else {
        for (size_t i = 0; i < materials.size(); i++) {
            meshFaces[i].setTexturePath(dirPath + materials[i].diffuse_texname);
        }
    }

    MeshData* meshData = new MeshData();

    bool nonTriMeshFound = false;

    // For each shape
    for (size_t i = 0; i < shapes.size(); i++) {
        size_t index_offset = 0;

        // For each face (ここではfaceが1ポリゴンにあたる)
        for (size_t f = 0; f < shapes[i].mesh.num_face_vertices.size(); f++) {
            int meshFaceIndex = shapes[i].mesh.material_ids[f];
            if( meshFaceIndex < 0 ) {
                meshFaceIndex = 0;
            }
            
            size_t fnum = shapes[i].mesh.num_face_vertices[f];

            // For each vertex in the face (triangulateするとfnumは3になる)
            for (size_t vi = 0; vi < fnum; vi++) {
                tinyobj::index_t idx = shapes[i].mesh.indices[index_offset + vi];

                float vx = attrib.vertices[3 * idx.vertex_index + 0];
                float vy = attrib.vertices[3 * idx.vertex_index + 1];
                float vz = attrib.vertices[3 * idx.vertex_index + 2];
                float nx = idx.normal_index >= 0
                    ? attrib.normals[3 * idx.normal_index + 0]
                    : 0.0f;
                float ny = idx.normal_index >= 0
                    ? attrib.normals[3 * idx.normal_index + 1]
                    : 0.0f;
                float nz = idx.normal_index >= 0
                    ? attrib.normals[3 * idx.normal_index + 2]
                    : 0.0f;
                float u = idx.texcoord_index >= 0
                    ? attrib.texcoords[2 * idx.texcoord_index + 0]
                    : 0.0f;
                float v = idx.texcoord_index >= 0
                    ? attrib.texcoords[2 * idx.texcoord_index + 1]
                    : 0.0f;
                
                meshFaces[meshFaceIndex].addVertex(idx.vertex_index,
                                                   idx.normal_index,
                                                   idx.texcoord_index,
                                                   vx,
                                                   vy,
                                                   vz,
                                                   nx,
                                                   ny,
                                                   nz,
                                                   u,
                                                   v);
            }
            
            if(fnum == 3) {
                const std::vector<tinyobj::index_t>& indices = shapes[i].mesh.indices;
                tinyobj::index_t idx0 = indices[index_offset];
                float x0 = attrib.vertices[3 * idx0.vertex_index + 0];
                float y0 = attrib.vertices[3 * idx0.vertex_index + 1];
                float z0 = attrib.vertices[3 * idx0.vertex_index + 2];

                tinyobj::index_t idx1 = indices[index_offset+1];
                float x1 = attrib.vertices[3 * idx1.vertex_index + 0];
                float y1 = attrib.vertices[3 * idx1.vertex_index + 1];
                float z1 = attrib.vertices[3 * idx1.vertex_index + 2];

                tinyobj::index_t idx2 = indices[index_offset+2];
                float x2 = attrib.vertices[3 * idx2.vertex_index + 0];
                float y2 = attrib.vertices[3 * idx2.vertex_index + 1];
                float z2 = attrib.vertices[3 * idx2.vertex_index + 2];
                
                meshData->addCollisionTriangle(x0, y0, z0,
                                               x1, y1, z1,
                                               x2, y2, z2);
            } else {
                nonTriMeshFound = true;
            }

            index_offset += fnum;
        }
    }
    
    if(nonTriMeshFound) {
        printf("Mesh was not triangulated\n");
    }
    
    for(int i=0; i<meshFaces.size(); ++i) {
        const string& texturePath = meshFaces[i].getTexturePath();
        MeshFaceData* meshFaceData = meshFaces[i].toMeshFaceData();
        meshData->addMeshFace(meshFaceData, texturePath);
    }

    return meshData;
}

static std::string getDirPath(const std::string& filePath) {
    size_t pos = filePath.find_last_of("/");
    return (std::string::npos == pos)
        ? ""
        : filePath.substr(0, pos+1);
}

/**
 * <!--  import():  -->
 */
MeshData* ObjImporter::import(const char* path) {
    std::string dirPath = getDirPath(path);
    const char* basePath = dirPath.empty()
        ? NULL
        : dirPath.c_str();

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string err;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err,
                                path, basePath, true);

    if (!err.empty()) {
        std::cerr << err << std::endl;
    }

    if (!ret) {
        printf("Failed to load/parse .obj.\n");
        return nullptr;
    }

    return loadObjMeshData(attrib, shapes, materials, dirPath);
}
