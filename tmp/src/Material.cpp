#include "Material.h"
#include "Texture.h"
#include "Shader.h"
#include "Matrix4f.h"
#include "MeshFaceData.h"

/**
 * <!--  draw():  -->
 */
void Material::draw(const MeshFaceData& meshFaceData,
					const Matrix4f& modelViewMat,
					const Matrix4f& modelViewProjectionMat) {
	texture->bind();
	
	shader->use();
	shader->beginRender(meshFaceData.getVertices());

	shader->setMatrix(modelViewMat, modelViewProjectionMat);
		
	glActiveTexture(GL_TEXTURE0);
	
	shader->render(meshFaceData.getIndices(),
				   meshFaceData.getIndicesSize());
	shader->endRender();
}

