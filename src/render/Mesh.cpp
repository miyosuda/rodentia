#include "Mesh.h"

#include <float.h>

#include "MeshFace.h"
#include "Matrix4f.h"


Mesh::~Mesh() {
    int size = meshFaces.size();
    for(int i=0; i<size; ++i) {
        delete meshFaces[i];
    }
    meshFaces.clear();
}

void Mesh::addMeshFace(MeshFace* meshFace) {
    meshFaces.push_back(meshFace);
    boundingBox.merge(meshFace->getBoundingBox());
}

void Mesh::draw(const RenderingContext& context) const {
    int size = meshFaces.size();
    for(int i=0; i<size; ++i) {
        meshFaces[i]->draw(context);
    }   
}

void Mesh::replaceMaterials(const vector<Material*>& materials) {
    for(unsigned int i=0; i<materials.size(); ++i) {
        if( i < meshFaces.size() ) {
            Material* material = materials[i];
            meshFaces[i]->replaceMaterial(material);
        }
    }
}
