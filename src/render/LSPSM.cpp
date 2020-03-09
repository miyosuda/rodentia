#include "LSPSM.h"
#include <float.h>
#include <math.h>

#include "BoundingBox.h"

#define NEW_FORMULA 1

static void createLookTo(const Vector3f& fromPos, const Vector3f& dir,
                         const Vector3f& up,
                         Matrix4f& viewMat) {
    Vector3f zaxis;
    zaxis.set(dir);
    zaxis *= -1.0f; // inverted
    zaxis.normalize();

    Vector3f xaxis;
    xaxis.cross(up, zaxis);
    xaxis.normalize();

    Vector3f yaxis;
    yaxis.cross(zaxis, xaxis);
    yaxis.normalize();

    viewMat.m00 = xaxis.x;
    viewMat.m10 = yaxis.x;
    viewMat.m20 = zaxis.x;
    viewMat.m30 = 0.0f;

    viewMat.m01 = xaxis.y;
    viewMat.m11 = yaxis.y;
    viewMat.m21 = zaxis.y;
    viewMat.m31 = 0.0f;
    
    viewMat.m02 = xaxis.z;
    viewMat.m12 = yaxis.z;
    viewMat.m22 = zaxis.z;
    viewMat.m32 = 0.0f;

    viewMat.m03 = -xaxis.dot(fromPos);
    viewMat.m13 = -yaxis.dot(fromPos);
    viewMat.m23 = -zaxis.dot(fromPos);
    viewMat.m33 = 1.0f;
}

static float getCrossingAngle(const Vector3f& v0,
                              const Vector3f& v1) {
    float d = v0.length() * v1.length();
    if( d == 0.0f ) {
        return 0.0f;
    }

    float c = v0.dot(v1) / d;
    if( c >= 1.0f ) {
        return 0.0;
    }
    if( c <= -1.0f ) {
        return M_PI;
    }
    return acosf(c);
}

static void transformCoord(const Vector3f& p, const Matrix4f& mat,
                           Vector3f& pout) {
    Vector4f tp;
    mat.transform(Vector4f(p.x, p.y, p.z, 1.0f), tp);
    pout.set(tp.x/tp.w, tp.y/tp.w, tp.z/tp.w);
}


//==============================
//     [VolumePoints Class]
//==============================

VolumePoints::VolumePoints() {
    init();
}

void VolumePoints::init() {
    points[0].set( -1.0f, +1.0f, -1.0f );
    points[1].set( -1.0f, -1.0f, -1.0f );
    points[2].set( +1.0f, -1.0f, -1.0f );
    points[3].set( +1.0f, +1.0f, -1.0f );
    points[4].set( -1.0f, +1.0f, +1.0f );
    points[5].set( -1.0f, -1.0f, +1.0f );
    points[6].set( +1.0f, -1.0f, +1.0f );
    points[7].set( +1.0f, +1.0f, +1.0f );
}

void VolumePoints::transform(const Matrix4f& matrix) {
    for(int i=0; i<POINT_SIZE; ++i) {
        Vector3f& point = points[i];
        transformCoord(point, matrix, point);
    }
}

void VolumePoints::computeBoundingBox(BoundingBox& boundingBox) const {
    for( int i=0; i<POINT_SIZE; ++i ) {
        const Vector3f& p = points[i];
        boundingBox.mergeVertex(p.x, p.y, p.z);
    }
}

VolumePoints& VolumePoints::operator=(const VolumePoints &value) {
    for( int i=0; i<POINT_SIZE; ++i ) {
        points[i].set(value.points[i]);
    }
    return (*this);
}

void VolumePoints::debugDump() const {
    for( int i=0; i<POINT_SIZE; ++i ) {
        points[i].debugDump();
    }
}


//==============================
//           [LSPSM]
//==============================

/**
 * <!--  LSPSM():  -->
 */
LSPSM::LSPSM()
    :
    nearClip(0.1f) {
}

/**
 * <!--  computeUpVector():  -->
 *
 * Calculate up vector with view and light vector.
 *
 * (This vector is on the place make by view and light vector and is orthogonal to 
 * light vector.)
 */
void LSPSM::computeUpVector(const Vector3f& viewDir,
                            const Vector3f& lightDir,
                            Vector3f& up) {
    Vector3f left;
    left.cross(lightDir, viewDir);
    up.cross(left, lightDir);
    up.normalize();
}

/**
 * <!--  computeMatrix_USM():  -->
 * 
 * Calculate matrix for plain shadow mapping.
 */
void LSPSM::computeMatrix_USM(VolumePoints& points) {
    // Calculate light view matrix. (TODO: lightDir is viewDir is parallel?)
    createLookTo( eyePosition, lightDir, viewDir, lightViewMat );

    // Convert volume points from world coordinate to light space)
    points.transform(lightViewMat);

    // Calculate AABB
    BoundingBox boundingBox;
    points.computeBoundingBox(boundingBox);

    // Adjust region.
    getUnitCubeClipMatrix(boundingBox, lightProjMat);
}

/**
 * <!--  computeMatrix_LSPSM():  -->
 *
 * Calculate light space projection shadow mapping matrix.
 */
void LSPSM::computeMatrix_LSPSM(float angle, VolumePoints& points) {
    // Store copy of points. (Now points stores view frustum in world coordinate.)
    VolumePoints pointsClone = points;

    //float sinGamma = sqrtf(1.0f - angle * angle); // original
    float sinGamma = sinf(fabsf(angle));

    // Calculate up vector.
    // (This vector is on the place by light vector and view vecetor and it is orthogonal to light vector)
    Vector3f up;
    computeUpVector(viewDir, lightDir, up);

    // Calclate light view matrix.
    // (Inverse matrix that originates from eye pos and looks to light dir)
    createLookTo(eyePosition, lightDir, up, lightViewMat);

    // Transform points with it and calculate AABB.
    points.transform(lightViewMat);

    BoundingBox boundingBox;
    points.computeBoundingBox(boundingBox);
    
    Vector3f halfExtent;
    boundingBox.getHalfExtent(halfExtent);

    // Calculate new frustum.
    const float factor = 1.0f / sinGamma;
    const float z_n = factor * nearClip;
    const float d = fabsf(halfExtent.y * 2.0f);
    
#if NEW_FORMULA
    const float z0 = - z_n;
    const float z1 = - ( z_n + d * sinGamma );
    const float n = d / ( sqrtf( z1 / z0 ) - 1.0f );
#else
    const float z_f = z_n + d * sinGamma;
    const float n = ( z_n + sqrtf( z_f * z_n ) ) * factor;
#endif

    const float f = n + d;

    // pos = eyePosition - up * (n - nearClip)
    Vector3f upScaled;
    upScaled.scale(n - nearClip, up);
    
    Vector3f pos;
    pos.sub(eyePosition, upScaled);

    // Light view for shadow mapping.
    // Originate from pos and looking to light direction.
    createLookTo(pos, lightDir, up, lightViewMat);

    // Calculate projection matrix with y diection.
    Matrix4f yProjMat;
    getPerspective(n, f, yProjMat);

    // Transform volume points from world coordinate to projected light space.
    Matrix4f tmpLightViewProjMat;
    tmpLightViewProjMat.mul(yProjMat, lightViewMat);
    pointsClone.transform(tmpLightViewProjMat);

    // Calculate AABB
    BoundingBox boundingBox2;
    pointsClone.computeBoundingBox(boundingBox2);

    // Adjust region
    Matrix4f clipMat;
    getUnitCubeClipMatrix(boundingBox2, clipMat);

    // Calculate projection matrix for shadow mapping.
    lightProjMat.mul(clipMat, yProjMat);
}

/**
 * <!--  updateShadowMatrix():  -->
 *
 * ライトの行列を更新する
 */
void LSPSM::updateShadowMatrix() {
    Matrix4f viewProjMat;
    viewProjMat.mul(eyeProjMat, eyeViewMat);

    // Calculate view volume.
    VolumePoints points;
    computeLightVolumePoints(viewProjMat, points);
    
    // Now points are corners of view frustum.
    
    // Calculate angle between view vector and light direction.
    float angle = getCrossingAngle(viewDir, lightDir);
    
    // If angle is 0 or 180 degree.
    if(angle == 0.0f || angle == M_PI) {
        // There is now skew with light, so use normal shadow mapping.
        computeMatrix_USM(points);
    } else {
        // Light Space Perspective Shadow Map
        computeMatrix_LSPSM(angle, points);
    }

    // Convert to left hand coordinate.
    Matrix4f scaleMat;
    scaleMat.setIdentity();
    scaleMat.m22 = -1.0f;
    lightProjMat.mul(scaleMat, lightProjMat);
    
    // Calculate result matrix for shadow mapping.
    lightViewProjMat.mul(lightProjMat, lightViewMat);
}

/**
 * <!--  computeLightVolumePoints():  -->
 */
void LSPSM::computeLightVolumePoints(const Matrix4f& viewProjMat, VolumePoints& points) {
    // Calculate view frustum
    computeViewFrustum(viewProjMat, points);

    // TODO: Include objects that cast its shadow into view frustum.
}

/**
 * <!--  computeViewFrustum():  -->
 *
 * Calculate camera view frustum.
 */
void LSPSM::computeViewFrustum(const Matrix4f& viewProjMat,
                               VolumePoints& points) {
    // Calc invrert matrix of View -> Projection.
    Matrix4f invViewProjMat;
    invViewProjMat.invert(viewProjMat);

    // Transform cube with invert matrix to get view frustum volume.
    points.transform(invViewProjMat);
}

/**
 * <!--  getUnitCubeClipMatrix():  -->
 *
 * Calculate matrix to adjust region.
 */
void LSPSM::getUnitCubeClipMatrix(const BoundingBox& boundingBox,
                                  Matrix4f& mat) const {
    Vector3f halfExtent;
    Vector3f center;
    const Vector3f& min = boundingBox.getMinPos();
    boundingBox.getHalfExtent(halfExtent);
    boundingBox.getCenter(center);
    
    float sx =  1.0f / halfExtent.x;
    float sy =  1.0f / halfExtent.y;
    float sz =  1.0f / (halfExtent.z * 2.0f); // TODO: check
    float tx = -center.x / halfExtent.x;
    float ty = -center.y / halfExtent.y;
    float tz = -min.z / (halfExtent.z * 2.0f); // TODO: check

    mat.m00 = sx;    mat.m01 = 0.0f;  mat.m02 = 0.0f;  mat.m03 = tx;
    mat.m10 = 0.0f;  mat.m11 = sy;    mat.m12 = 0.0f;  mat.m13 = ty;
    mat.m20 = 0.0f;  mat.m21 = 0.0f;  mat.m22 = sz;    mat.m23 = tz;
    mat.m30 = 0.0f;  mat.m31 = 0.0f;  mat.m32 = 0.0f;  mat.m33 = 1.0f;
}

/**
 * <!--  getPerspective():  -->
 *
 * Calc projection matrix for Y direction.
 */
void LSPSM::getPerspective(const float near, const float far, Matrix4f& mat) const {
    mat.setIdentity();
    mat.m11 = far / (far - near);
    mat.m31 = 1.0f;
    mat.m13 = -far * near / (far - near);
    mat.m33 = 0.0f;
}
