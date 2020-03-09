#include "MeshData.h"
#include "Mesh.h"
#include "MeshFace.h"
#include "MeshFaceData.h"
#include "TextureManager.h"
#include "ShaderManager.h"
#include "Material.h"

/**
 * <!--  ~MeshData():  -->
 */
MeshData::~MeshData() {
    for(int i=0; i<meshFaceDatas.size(); ++i) {
        delete meshFaceDatas[i];
    }
    meshFaceDatas.clear();
}

/**
 * <!--  addMeshFace():  -->
 */
void MeshData::addMeshFace(MeshFaceData* meshFaceData, const string& texturePath) {
    meshFaceDatas.push_back(meshFaceData);
    texturePathes.push_back(texturePath);
}

/**
 * <!--  toMesh():  -->
 */
Mesh* MeshData::toMesh(Material* material) {
    // Ignoring texturePathes in this object and using material argument instead.
    
    Mesh* mesh = new Mesh();

    for(int i=0; i<meshFaceDatas.size(); ++i) {
        MeshFace* meshFace = new MeshFace(material,
                                          *meshFaceDatas[i]);
        mesh->addMeshFace(meshFace);
    }

    return mesh;
}

/**
 * <!--  toMesh():  -->
 */
Mesh* MeshData::toMesh(TextureManager& textureManager, ShaderManager& shaderManager) {
    Mesh* mesh = new Mesh();

    for(int i=0; i<meshFaceDatas.size(); ++i) {
        const string& texturePath = texturePathes[i];
        Texture* texture;
        if( texturePath.empty() ) {
            texture = textureManager.getColorTexture(1.0f, 1.0f, 1.0f);
        } else {
            texture = textureManager.loadTexture(texturePath.c_str());
            if( texture == nullptr ) {
                delete mesh;
                return nullptr;
            }
        }
        
        Shader* shader = shaderManager.getDiffuseShader();
        Shader* shadowDepthShader = shaderManager.getShadowDepthShader();
        
        Material* material = new Material(texture, shader, shadowDepthShader);
        MeshFace* meshFace = new MeshFace(material, *meshFaceDatas[i]);
        
        mesh->addMeshFace(meshFace);
    }
    
    return mesh;
}
