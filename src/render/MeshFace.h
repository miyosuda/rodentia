// -*- C++ -*-
#ifndef MESHFACE_HEADER
#define MESHFACE_HEADER

class Material;
class Matrix4f;
class MeshFaceData;

class MeshFace {
private:
	Material* material;
	const MeshFaceData& meshFaceData;

public:
	MeshFace( Material* material_,
			  const MeshFaceData& meshFaceData );
	~MeshFace();
	void draw( const Matrix4f& modelViewMat, 
			   const Matrix4f& projectionMat );

	const MeshFaceData& debugGetMeshFaceData() const {
		return meshFaceData;
	}
};

#endif
