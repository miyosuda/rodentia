#include "BoundingBox.h"
#include "Matrix4f.h"

#include <gtest/gtest.h>
#include <math.h>

namespace {
    class BoundingBoxTest : public ::testing::Test {

        
    protected:
        void transformBoundingBoxLazySub(const Matrix4f& mat,
                                         const Vector3f& pos,
                                         BoundingBox& outBoundingBox) {
            Vector4f pos_(pos.x, pos.y, pos.z, 1.0f);
            Vector4f transformedPos;
            mat.transform(pos_, transformedPos);
            outBoundingBox.mergeVertex(transformedPos.x, transformedPos.y, transformedPos.z);
        }
        
        void transformBoundingBoxLazy(const Matrix4f& mat,
                                      const Vector3f& minPos,
                                      const Vector3f& maxPos,
                                      BoundingBox& outBoundingBox) {
            transformBoundingBoxLazySub( mat, Vector3f(minPos.x, minPos.y, maxPos.z), outBoundingBox);
            transformBoundingBoxLazySub( mat, Vector3f(minPos.x, maxPos.y, minPos.z), outBoundingBox);
            transformBoundingBoxLazySub( mat, Vector3f(minPos.x, maxPos.y, maxPos.z), outBoundingBox);
            transformBoundingBoxLazySub( mat, Vector3f(maxPos.x, minPos.y, minPos.z), outBoundingBox);
            transformBoundingBoxLazySub( mat, Vector3f(maxPos.x, minPos.y, maxPos.z), outBoundingBox);
            transformBoundingBoxLazySub( mat, Vector3f(maxPos.x, maxPos.y, minPos.z), outBoundingBox);
            transformBoundingBoxLazySub( mat, Vector3f(maxPos.x, maxPos.y, maxPos.z), outBoundingBox);
            transformBoundingBoxLazySub( mat, Vector3f(minPos.x, minPos.y, minPos.z), outBoundingBox);
        }
    };      
    
    TEST_F(BoundingBoxTest, transform) {
        BoundingBox boundingBox;

        boundingBox.mergeVertex(-1.0f, -1.0f, -1.0f);
        boundingBox.mergeVertex( 1.0f,  1.0f,  1.0f);

        Matrix4f mat;
        mat.setRotationZ(M_PI / 3.0f);

        float scaleX = 2.0f;
        float scaleY = 3.0f;
        float scaleZ = 10.0f;

        BoundingBox transformedBoundingBox;
        boundingBox.transform(scaleX, scaleY, scaleZ,
                              mat,
                              transformedBoundingBox);

        Vector3f center;
        Vector3f halfExtent;

        transformedBoundingBox.getCenter(center);
        transformedBoundingBox.getHalfExtent(halfExtent);

        // Transform manually with lazy way.
        BoundingBox transformedBoundingBox_;

        transformBoundingBoxLazy(mat,
                                 Vector3f(-scaleX, -scaleY, -scaleZ),
                                 Vector3f( scaleX,  scaleY,  scaleZ),
                                 transformedBoundingBox_);

        Vector3f center_;
        Vector3f halfExtent_;
        transformedBoundingBox_.getCenter(center_);
        transformedBoundingBox_.getHalfExtent(halfExtent_);
        
        const float EPSILON = 0.00001f;
        
        EXPECT_NEAR(center.x, center_.x, EPSILON);
        EXPECT_NEAR(center.y, center_.y, EPSILON);
        EXPECT_NEAR(center.z, center_.z, EPSILON);
        
        EXPECT_NEAR(halfExtent.x, halfExtent_.x, EPSILON);
        EXPECT_NEAR(halfExtent.y, halfExtent_.y, EPSILON);
        EXPECT_NEAR(halfExtent.z, halfExtent_.z, EPSILON);
    }
}
