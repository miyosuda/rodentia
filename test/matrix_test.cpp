#include "Matrix4f.h"
#include <gtest/gtest.h>

namespace {
    class Matrix4fTest : public ::testing::Test {
    protected:
        void checkMatrixNear(const Matrix4f& mat0, const Matrix4f& mat1) {
            const float* fmat0 = mat0.getPointer();
            const float* fmat1 = mat1.getPointer();
        
            for(int i=0; i<16; ++i) {
                EXPECT_NEAR(fmat0[i], fmat1[i], 0.001f);
            }
        }
    };

    TEST_F(Matrix4fTest, invert) {
        Matrix4f mat0;
        mat0.set(1.0f, 2.0f, 3.0f, 4.0f,
                 0.0f, 1.0f, 0.0f, 0.0f,
                 0.0f, 1.0f, 2.0f, 3.0f,
                 1.0f, 0.0f, 3.0f, 1.0f);

        Matrix4f mat1(mat0);

        Matrix4f tmpMat;
        mat0.invert(tmpMat);
        tmpMat.invert(mat0);
        
        checkMatrixNear(mat1, mat0);
    }   

    TEST_F(Matrix4fTest, invert2) {
        Matrix4f mat0;

        mat0.set(1.0f, 2.0f, 3.0f, 4.0f,
                 0.0f, 1.0f, 0.0f, 0.0f,
                 0.0f, 1.0f, 2.0f, 3.0f,
                 1.0f, 0.0f, 3.0f, 1.0f);

        Matrix4f mat1(mat0);
        
        mat0.invert();
        mat0.invert();

        checkMatrixNear(mat1, mat0);
    }

    TEST_F(Matrix4fTest, invertRT) {
        Matrix4f mat0;

        mat0.setRotationX(1.0f);
        mat0.setColumn(3, Vector4f(1.0f, 2.0f, 3.0f, 1.0f));

        Matrix4f mat1(mat0);

        mat0.invertRT();
        mat0.invertRT();

        checkMatrixNear(mat1, mat0);
    }

    TEST_F(Matrix4fTest, invertRT2) {
        Matrix4f mat0;

        mat0.setRotationX(1.0f);
        mat0.setColumn(3, Vector4f(1.0f, 2.0f, 3.0f, 1.0f));

        Matrix4f mat1(mat0);
        Matrix4f mat2;
        Matrix4f mat3;

        mat2.invertRT(mat0);
        mat3.invertRT(mat2);

        checkMatrixNear(mat1, mat3);
    }

    TEST_F(Matrix4fTest, invertR) {
        Matrix4f mat0;

        mat0.setRotationY(1.0f);

        Matrix4f mat1(mat0);

        mat0.invertRT();
        mat0.invertRT();

        checkMatrixNear(mat1, mat0);
    }

    TEST_F(Matrix4fTest, invertR2) {
        Matrix4f mat0;

        mat0.setRotationZ(1.0f);

        Matrix4f mat1(mat0);
        Matrix4f mat2;
        Matrix4f mat3;

        mat2.invertR(mat0);
        mat3.invertR(mat2);

        checkMatrixNear(mat1, mat3);
    }

    TEST_F(Matrix4fTest, quaternion) {
        Matrix4f mat0;
        mat0.setRotationX(1.0f);
        Matrix4f mat1(mat0);

        Quat4f q;
        q.set(mat0);

        Matrix4f mat2;
        mat2.set(q);

        checkMatrixNear(mat1, mat2);
    }
}
