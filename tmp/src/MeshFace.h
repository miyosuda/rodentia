// -*- C++ -*-
#ifndef MESHFACE_HEADER
#define MESHFACE_HEADER

class Material;
class Matrix4f;

class MeshFace {
private:
	Material* material;

	float* vertices;
	int verticesSize;
	short* indices;
	int indicesSize;

public:
	MeshFace( Material* material_,
			  float* vertices_,
			  int verticesSize_,
			  short* indices_,
			  int indicesSize_ );
	~MeshFace();
	void draw( const Matrix4f& modelViewMat, 
			   const Matrix4f& projectionMat );
};

#endif
