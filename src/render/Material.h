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

public:
	Material(Texture* texture_,
			 Shader* shader_)
		:
		texture(texture_),
		shader(shader_) {
	}

	void draw(const MeshFaceData& meshFaceData,
			  const RenderingContext& context);
};

#endif
