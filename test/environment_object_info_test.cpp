#include "EnvironmentObject.h"
#include "Matrix4f.h"
#include "Vector3f.h"

#include <gtest/gtest.h>

namespace {
	class EnvironmentObjectInfoTest : public ::testing::Test {
	};
		
	TEST_F(EnvironmentObjectInfoTest, set) {
		EnvironmentObjectInfo info;

		Matrix4f mat;
		mat.setIdentity();

		Vector4f pos(100.0f, 200.0f, 300.0f, 1.0f);
		mat.setColumn(3, pos);

		Vector3f velocity(1.0f, 2.0f, 3.0f);

		info.set(mat, velocity);

		const float EPSILON = 0.00001f;

		// Check pos
		EXPECT_NEAR(info.pos.x, pos.x, EPSILON);
		EXPECT_NEAR(info.pos.y, pos.y, EPSILON);
		EXPECT_NEAR(info.pos.z, pos.z, EPSILON);

		// Check velocity
		EXPECT_NEAR(info.velocity.x, velocity.x, EPSILON);
		EXPECT_NEAR(info.velocity.y, velocity.y, EPSILON);
		EXPECT_NEAR(info.velocity.z, velocity.z, EPSILON);

        // TODO:

        /*
		// Check euler angles

		// Rotation X
		mat.setRotationX(1.0f);
		info.set(mat, velocity);

		EXPECT_NEAR(info.rot.x, 1.0f, EPSILON);
		EXPECT_NEAR(info.rot.y, 0.0f, EPSILON);
		EXPECT_NEAR(info.rot.z, 0.0f, EPSILON);
        EXPECT_NEAR(info.rot.w, 0.0f, EPSILON);

		// Rotation Y
		mat.setRotationY(1.0f);
		info.set(mat, velocity);
		
		EXPECT_NEAR(info.eulerAngles.x, 0.0f, EPSILON);
		EXPECT_NEAR(info.eulerAngles.y, 1.0f, EPSILON);
		EXPECT_NEAR(info.eulerAngles.z, 0.0f, EPSILON);

		// Rotation Z
		mat.setRotationZ(1.0f);
		info.set(mat, velocity);
		
		EXPECT_NEAR(info.eulerAngles.x, 0.0f, EPSILON);
		EXPECT_NEAR(info.eulerAngles.y, 0.0f, EPSILON);
		EXPECT_NEAR(info.eulerAngles.z, 1.0f, EPSILON);
        */
	}
}
