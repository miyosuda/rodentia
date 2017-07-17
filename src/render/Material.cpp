#include "Material.h"
#include "Texture.h"
#include "Shader.h"
#include "Matrix4f.h"
#include "MeshFaceData.h"

/**
 * <!--  draw():  -->
 */
void Material::draw(const MeshFaceData& meshFaceData,
					const Matrix4f& modelMat,
					const Matrix4f& modelViewMat,
					const Matrix4f& modelViewProjectionMat) {

	if( texture != nullptr ) {
		texture->bind();
		glActiveTexture(GL_TEXTURE0);
	}

	shader->use();
	shader->beginRender(meshFaceData.getVertices());

	shader->setMatrix(modelMat, modelViewMat, modelViewProjectionMat);

	shader->render(meshFaceData.getIndices(),
				   meshFaceData.getIndicesSize());

	shader->endRender();
}
