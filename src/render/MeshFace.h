// -*- C++ -*-
#ifndef MESHFACE_HEADER
#define MESHFACE_HEADER

class Material;
class Matrix4f;
class Vector3f;
class MeshFaceData;
class RenderingContext;


class MeshFace {
private:
	Material* material;
	const MeshFaceData& meshFaceData;

public:
	MeshFace( Material* material_,
			  const MeshFaceData& meshFaceData );
	~MeshFace();
	void draw(const RenderingContext& context);
	void calcBoundingBox(Vector3f& minPos, Vector3f& maxPos) const;

	const MeshFaceData& debugGetMeshFaceData() const {
		return meshFaceData;
	}
};

#endif
