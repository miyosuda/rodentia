#include "Material.h"
#include "Texture.h"
#include "Shader.h"
#include "Matrix4f.h"
#include "MeshFaceData.h"
#include "RenderingContext.h"

/**
 * <!--  draw():  -->
 */
void Material::draw(const MeshFaceData& meshFaceData,
					const RenderingContext& context) {

	if( texture != nullptr ) {
		texture->bind();
		glEnable(GL_TEXTURE_2D);
		glActiveTexture(GL_TEXTURE0);
	} else {
		glDisable(GL_TEXTURE_2D);
	}

	shader->use();
	shader->setup(context);
	meshFaceData.draw();
}
