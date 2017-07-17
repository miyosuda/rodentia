// -*- C++ -*-
#ifndef MATERIAL_HEADER
#define MATERIAL_HEADER

class Texture;
class Shader;
class Matrix4f;
class MeshFaceData;

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
			  const Matrix4f& modelMat,
			  const Matrix4f& modelViewMat,
			  const Matrix4f& modelViewProjectionMat);
};

#endif
