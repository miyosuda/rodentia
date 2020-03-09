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

    if( context.isRenderingShadow() ) {
        glDisable(GL_TEXTURE_2D);
        
        shadowDepthShader->use();
        shadowDepthShader->setup(context);
        meshFaceData.draw(true);
    } else {
        if( texture != nullptr ) {
            glEnable(GL_TEXTURE_2D);
            glActiveTexture(GL_TEXTURE0);
            texture->bind();
        } else {
            glDisable(GL_TEXTURE_2D);
        }
        
        shader->use();
        shader->setup(context);
        meshFaceData.draw(false);
    }
}
