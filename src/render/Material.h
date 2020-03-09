// -*- C++ -*-
#ifndef MATERIAL_HEADER
#define MATERIAL_HEADER

class Texture;
class Shader;
class Matrix4f;
class MeshFaceData;
class RenderingContext;


class Material {
private:
    Texture* texture;
    Shader* shader;
    Shader* shadowDepthShader;

public:
    Material(Texture* texture_,
             Shader* shader_,
             Shader* shadowDepthShader_)
        :
        texture(texture_),
        shader(shader_),
        shadowDepthShader(shadowDepthShader_) {
    }

    void draw(const MeshFaceData& meshFaceData,
              const RenderingContext& context);
};

#endif
