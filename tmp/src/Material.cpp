#include "Material.h"
#include "Texture.h"
#include "Shader.h"
#include "Matrix4f.h"

/**
 * <!--  draw():  -->
 */
void Material::draw(float* vertices,
					short* indices,
					int indicesSize,
					const Matrix4f& modelViewMat,
					const Matrix4f& modelViewProjectionMat) {
	texture->bind();
	shader->use();
	shader->beginRender(vertices);

	shader->setMatrix(modelViewMat, modelViewProjectionMat);
		
	glActiveTexture(GL_TEXTURE0);
		
	shader->render(indices, indicesSize);
	shader->endRender();
}
