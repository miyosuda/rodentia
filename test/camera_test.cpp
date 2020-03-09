#include <gtest/gtest.h>

#include "Camera.h"
#include "Vector3f.h"
#include "Vector4f.h"

namespace {
    class CameraTest : public ::testing::Test {
    protected:
        void checkMatrixNear(const Matrix4f& mat0, const Matrix4f& mat1) {
            const float* fmat0 = mat0.getPointer();
            const float* fmat1 = mat1.getPointer();
            
            for(int i=0; i<16; ++i) {
                EXPECT_NEAR(fmat0[i], fmat1[i], 0.001f);
            }
        }

        void checkVector3Near(const Vector3f& vec0, const Vector3f& vec1) {
            EXPECT_NEAR(vec0.x, vec1.x, 0.001f);
            EXPECT_NEAR(vec0.y, vec1.y, 0.001f);
            EXPECT_NEAR(vec0.z, vec1.z, 0.001f);
        }
    };

    TEST_F(CameraTest, lookAt) {
        const Vector3f fromPos(0.0f, 0.0f, 0.0f);
        const Vector3f toPos(0.0f, 0.0f, -1.0f);
        const Vector3f up(0.0f, 1.0f, 0.0f);

        Camera camera;
        camera.lookAt(fromPos, toPos, up);

        // This mat should be identity matrix.
        const Matrix4f& mat = camera.getMat();
        
        Matrix4f identityMat;
        identityMat.setIdentity();
        
        checkMatrixNear(identityMat, mat);
    }

    TEST_F(CameraTest, initPerspective) {
        const float znear = 1.0f;
        const float zfar = 10.0f;
        const float focalLength = 35.0f;
        const float ratio = 1.0f;
        
        Camera camera;
        camera.initPerspective(znear, zfar, focalLength, ratio, false);
        
        const Matrix4f& projMat = camera.getProjectionMat();

        Vector4f p;
        
        // Near plance center should be projected to (0,0,-1)
        Vector4f pnear(0.0f, 0.0f, -znear, 1.0f);
        projMat.transform(pnear, p);
        Vector3f pnear_(p.x/p.w, p.y/p.w, p.z/p.w);
        checkVector3Near(pnear_, Vector3f(0.0f, 0.0f, -1.0f));
        
        // Far plance center should be projected to (0,0,1)
        Vector4f pfar(0.0f, 0.0f, -zfar, 1.0f);
        projMat.transform(pfar, p);
        Vector3f pfar_(p.x/p.w, p.y/p.w, p.z/p.w);
        checkVector3Near(pfar_, Vector3f(0.0f, 0.0f, 1.0f));
    }
}
