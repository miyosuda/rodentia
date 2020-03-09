#include "RenderingContext.h"
#include "BoundingBox.h"


/**
 * <!--  RenderingContext():  -->
 */
RenderingContext::RenderingContext()
    :
    depthBiasMat(
        0.5f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.5f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.5f, 0.0f,
        0.5f, 0.5f, 0.5f, 1.0f)
{
    setPath(SHADOW);
    setLight(Vector3f(-0.5f, -1.0f, -0.4f), // lightDir
             Vector3f(1.0f, 1.0f, 1.0f), // lightColor
             Vector3f(0.4f, 0.4f, 0.4f), // ambientColor
             0.2f); // shadowColorRate

    cameraInvMat.setZero();
    cameraProjectionMat.setZero();
}

/**
 * <!--  setPath():  -->
 */
void RenderingContext::setPath(Path path_) {
    path = path_;
}

/**
 * <!--  setModelMat():  -->
 */
void RenderingContext::setModelMat(Matrix4f modelMat_) {
    modelMat.set(modelMat_);

    const Matrix4f& depthViewProjectionMat = lspsm.getLightViewProjection();
    depthModelViewProjectionMat.mul(depthViewProjectionMat, modelMat);

    if( !isRenderingShadow() ) {
        // Set matrix for normal rendering
        const Matrix4f& viewMat = cameraInvMat;
        const Matrix4f& projectionMat = cameraProjectionMat;

        modelViewMat.mul(viewMat, modelMat);
        modelViewProjectionMat.mul(projectionMat, modelViewMat);
        
        depthBiasModelViewProjectionMat.mul(depthBiasMat, depthModelViewProjectionMat);
    }
}

/**
 * <!--  setCamera():  -->
 */
void RenderingContext::setCamera(const Matrix4f& cameraMat,
                                 const Matrix4f& cameraInvMat_,
                                 const Matrix4f& cameraProjectionMat_) {
    // TODO: 複数の関数呼び出し整理できる
    cameraInvMat.set(cameraInvMat_);
    cameraProjectionMat.set(cameraProjectionMat_);
    
    const Vector4f& pos = cameraMat.getColumnRef(3);
    const Vector4f& zaxis = cameraMat.getColumnRef(2);

    Vector3f viewDir(-zaxis.x, -zaxis.y, -zaxis.z);

    lspsm.setNearClip(1.0f);
    
    lspsm.setViewDir(viewDir);
    lspsm.setLightDir(lightDir);

    lspsm.setEyeView(cameraInvMat);
    lspsm.setEyePos(Vector3f(pos.x, pos.y, pos.z));
    lspsm.setEyeProjection(cameraProjectionMat);
    lspsm.updateShadowMatrix();
}

/**
 * <!--  setLight():  -->
 */
void RenderingContext::setLight(const Vector3f& lightDir_,
                                const Vector3f& lightColor_,
                                const Vector3f& ambientColor_,
                                float shadowColorRate_) {
    lightDir.set(lightDir_);
    lightDir.normalize();

    lightColor.set(lightColor_.x, lightColor_.y, lightColor_.z, 1.0f);
    ambientColor.set(ambientColor_.x, ambientColor_.y, ambientColor_.z, 1.0f);
    shadowColorRate = shadowColorRate_;
}

/**
 * <!--  setBoundingBoxForShadow():  -->
 */
void RenderingContext::setBoundingBoxForShadow(const BoundingBox& boundingBox) {
    // TODO: リネームして、LiSPSMのBvolumeのclippingに使う.
}
