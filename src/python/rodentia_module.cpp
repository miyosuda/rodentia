#include <Python.h>

#include <vector>
#include <set>
#include <string>
using namespace std;

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "Environment.h"
#include "Action.h"
#include "Vector3f.h"
#include "Matrix4f.h"
#include "EnvironmentObject.h"

#define RODENTIA_MODULE_VERSION "0.0.8"


//---------------------------------------------------------
//                    [Interface]
//---------------------------------------------------------

static Environment* createEnvironment() {
    Environment* environment = new Environment();
    return environment;
}

static bool initEnvironment(Environment* environment) {
    if( !environment->init() ) {
        return false;
    }
    
    return true;
}

static int addCameraView(Environment* environment, int width, int height,
                         const Vector3f& bgColor,
                         float nearClip, float farClip, float focalLength,
                         int shadowBufferWidth) {
    int cameraId = environment->addCameraView(width, height, bgColor,
                                              nearClip, farClip, focalLength,
                                              shadowBufferWidth);
    return cameraId;
}

static int addAgent(Environment* environment,
                    float radius,
                    const Vector3f& pos,
                    float rotY,
                    float mass,
                    bool detectCollision,
                    const Vector3f& color) {
    return environment->addAgent(radius,
                                 pos, rotY,
                                 mass,
                                 detectCollision,
                                 color);
}

static void releaseEnvironment(Environment* environment) {
    environment->release();
    delete environment;
}

static void control(Environment* environment, int id, const Action& action) {
    environment->control(id, action);
}

static void applyImpulse(Environment* environment, int id, const Vector3f& impulse) {
    environment->applyImpulse(id, impulse);
}

static void step(Environment* environment, CollisionResult& collisionResult) {
    environment->step(collisionResult);
}

static void render(Environment* environment, int cameraId,
                   const Vector3f& pos,
                   const Quat4f& rot,
                   const set<int> ignoreIds) {
    environment->render(cameraId, pos, rot, ignoreIds);
}

static int addBox(Environment* environment,
                  const char* texturePath,
                  const Vector3f& color,
                  const Vector3f& halfExtent,
                  const Vector3f& pos,
                  const Quat4f& rot,
                  float mass,
                  bool detectCollision,
                  bool visible) {
    return environment->addBox(texturePath,
                               color,
                               halfExtent, pos, rot,
                               mass,
                               detectCollision,
                               visible);
}

static int addSphere(Environment* environment,
                     const char* texturePath,
                     const Vector3f& color,
                     float radius,
                     const Vector3f& pos,
                     const Quat4f& rot,
                     float mass,
                     bool detectCollision,
                     bool visible) {
    return environment->addSphere(texturePath,
                                  color,
                                  radius,
                                  pos, rot,
                                  mass,
                                  detectCollision,
                                  visible);
}

static int addModel(Environment* environment,
                    const char* path,
                    const Vector3f& color,
                    const Vector3f& scale,
                    const Vector3f& pos,
                    const Quat4f& rot,
                    float mass,
                    bool detectCollision,
                    bool useMeshCollision,
                    bool useCollisionFile,
                    bool visible) {
    return environment->addModel(path,
                                 color,
                                 scale, pos, rot, 
                                 mass,
                                 detectCollision,
                                 useMeshCollision,
                                 useCollisionFile,
                                 visible);
}

static void removeObj(Environment* environment, 
                      int id) {
    environment->removeObject(id);
}

static void locateObject(Environment* environment,
                         int id,
                         const Vector3f& pos,
                         const Quat4f& rot) {
    environment->locateObject(id, pos, rot);
}

static void locateAgent(Environment* environment,
                        int id,
                        const Vector3f& pos,
                        float rotY) {
    environment->locateAgent(id, pos, rotY);
}

static bool getObjectInfo(Environment* environment,
                          int id, EnvironmentObjectInfo& info) {
    return environment->getObjectInfo(id, info);
}

static void setLight(Environment* environment,
                     const Vector3f& dir,
                     const Vector3f& color,
                     const Vector3f& ambinetColor,
                     float shadowColorRate) {
    environment->setLight(dir, color, ambinetColor, shadowColorRate);
}

static int getActionSize(Environment* environment) {
    return Action::getActionSize();
}

static void replaceObjectTextures(Environment* environment,
                                  int id, const vector<string>& texturePathes) {
    environment->replaceObjectTextures(id, texturePathes);
}

//---------------------------------------------------------
//                     [Python funcs]
//---------------------------------------------------------

static bool checkArrayDim(PyArrayObject* array, int dim, const char* name) {
    if (PyArray_NDIM(array) != 1 ||
        PyArray_DIM(array, 0) != dim) {
        PyErr_Format(PyExc_ValueError, "%s must have shape (%d)", name, dim);
        return false;
    }
    return true;
}

static const float* getFloatArrayData(PyObject* obj, int dim, const char* name) {
    PyArrayObject* array = (PyArrayObject*)obj;

    if( !checkArrayDim(array, dim, name) ) {
        return nullptr;
    }

    if (PyArray_TYPE(array) != NPY_FLOAT) {
        PyErr_Format(PyExc_ValueError, "%s must have dtype np.float32", name);
        return nullptr;
    }

    return (const float*)PyArray_DATA(array);
}

static const int* getIntArrayData(PyObject* obj, int dim, const char* name) {
    PyArrayObject* array = (PyArrayObject*)obj;

    if( !checkArrayDim(array, dim, name) ) {
        return nullptr;
    }

    if (PyArray_TYPE(array) != NPY_INT) {
        PyErr_Format(PyExc_ValueError, "%s must have dtype np.int32", name);
        return nullptr;
    }
    
    return (const int*)PyArray_DATA(array);
}

typedef struct {
    PyObject_HEAD
    Environment* environment;
} EnvObject;

static void EnvObject_dealloc(EnvObject* self) {
    if( self->environment != nullptr ) {
        releaseEnvironment(self->environment);
        self->environment = nullptr;
    }

    (((PyObject*)(self))->ob_type)->tp_free((PyObject*)self);
}

static PyObject* EnvObject_new(PyTypeObject* type,
                               PyObject* args,
                               PyObject* kwds) {
    EnvObject* self;
    
    self = (EnvObject*)type->tp_alloc(type, 0);
    
    if (self != nullptr) {
        Environment* environment = createEnvironment();
        if (environment == nullptr) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create rodentia environment");
            Py_DECREF(self);
            return nullptr;
        }
        
        self->environment = environment;
    }
    
    return (PyObject*)self;
}

static int Env_init(EnvObject* self, PyObject* args, PyObject* kwds) {
    if (self->environment == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "rodentia environment not setup");
        return -1;
    }

    // Initialize environment
    if ( !initEnvironment(self->environment) ) {
        PyErr_Format(PyExc_RuntimeError, "Failed to init environment.");
        return -1;
    }

    return 0;
}

static PyObject* Env_release(EnvObject* self, PyObject* args, PyObject* kwds) {
    if (self->environment == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "rodentia environment not setup");
        return nullptr;
    }

    // Release environment
    releaseEnvironment(self->environment);

    // Set self environment null
    self->environment = nullptr;
    
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* Env_add_camera_view(EnvObject* self, PyObject* args, PyObject* kwds) { 
    const char *kwlist[] = { "width",
                             "height",
                             "bg_color",
                             "near",
                             "far",
                             "focal_length",
                             "shadow_buffer_width",
                             nullptr };

    // Get argument
    int width;
    int height;
    PyObject* bgColorObj = nullptr;
    float nearClip;
    float farClip;
    float focalLength;
    int shadowBufferWidth;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiO!fffi", const_cast<char**>(kwlist),
                                     &width,
                                     &height,
                                     &PyArray_Type, &bgColorObj,
                                     &nearClip,
                                     &farClip,
                                     &focalLength,
                                     &shadowBufferWidth)) {
        PyErr_SetString(PyExc_RuntimeError, "init argument shortage");
        return nullptr;
    }

    if (self->environment == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "rodentia environment not setup");
        return nullptr;
    }

    const float* bgColorArr = getFloatArrayData(bgColorObj, 3, "bg_color");
    if( bgColorArr == nullptr ) {
        return nullptr;
    }

    Vector3f bgColor = Vector3f(bgColorArr[0], bgColorArr[1], bgColorArr[2]);

    // Initialize environment
    int cameraId = addCameraView(self->environment, width, height, bgColor,
                                 nearClip, farClip, focalLength,
                                 shadowBufferWidth);
    if (cameraId < 0) {
        PyErr_Format(PyExc_RuntimeError, "Failed to init environment.");
        return nullptr;
    }

    // Returning object ID
    PyObject* idObj = PyLong_FromLong(cameraId);
    return idObj;
}

static PyObject* Env_add_agent(EnvObject* self, PyObject* args, PyObject* kwds) {
    float radius;
    PyObject* posObj = nullptr;
    float rotY;
    float mass;
    int detectCollision;
    PyObject* colorObj = nullptr;    

    // Get argument
    const char* kwlist[] = {"radius", "pos", "rot_y", "mass",
                            "detect_collision", "color",
                            nullptr};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "fO!ffiO!", const_cast<char**>(kwlist),
                                     &radius,
                                     &PyArray_Type, &posObj,
                                     &rotY,
                                     &mass,
                                     &detectCollision,
                                     &PyArray_Type, &colorObj)) {
        return nullptr;
    }
    
    if (self->environment == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "rodentia environment not setup");
        return nullptr;
    }

    // pos
    const float* posArr = getFloatArrayData(posObj, 3, "pos");
    if( posArr == nullptr ) {
        return nullptr;
    }

    Vector3f pos(posArr[0], posArr[1], posArr[2]);

    // color
    const float* colorArr = getFloatArrayData(colorObj, 3, "color");
    if( colorArr == nullptr ) {
        return nullptr;
    }
    Vector3f color(colorArr[0], colorArr[1], colorArr[2]);
    
    int id = addAgent(self->environment,
                      radius,
                      pos, rotY,
                      mass,
                      detectCollision != 0,
                      color);

    // Returning object ID
    PyObject* idObj = PyLong_FromLong(id);
    return idObj;
}

static PyObject* Env_control(EnvObject* self, PyObject* args, PyObject* kwds) {
    int id;
    PyObject* actionObj = nullptr;

    // Get argument
    const char* kwlist[] = {"id", "action", nullptr};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iO!", const_cast<char**>(kwlist),
                                     &id,
                                     &PyArray_Type, &actionObj)) {
        return nullptr;
    }
    
    if (self->environment == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "rodentia environment not setup");
        return nullptr;
    }

    // action
    int actionSize = getActionSize(self->environment);
    const int* actionArr = getIntArrayData(actionObj, actionSize, "action");
    if( actionArr == nullptr ) {
        return nullptr;
    }
    
    Action action(actionArr[0], actionArr[1], actionArr[2]);

    // Process control
    control(self->environment, id, action);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* Env_apply_impulse(EnvObject* self, PyObject* args, PyObject* kwds) {
    int id;
    PyObject* impulseObj = nullptr;

    // Get argument
    const char* kwlist[] = {"id", "impulse", nullptr};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iO!", const_cast<char**>(kwlist),
                                     &id,
                                     &PyArray_Type, &impulseObj)) {
        return nullptr;
    }
    
    if (self->environment == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "rodentia environment not setup");
        return nullptr;
    }

    // impulse
    const float* impulseArr = getFloatArrayData(impulseObj, 3, "impulse");
    if( impulseArr == nullptr ) {
        return nullptr;
    }
    Vector3f impulse(impulseArr[0], impulseArr[1], impulseArr[2]);    

    // Apply impulse
    applyImpulse(self->environment, id, impulseArr);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* Env_step(EnvObject* self, PyObject* args, PyObject* kwds) {
    if (self->environment == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "rodentia environment not setup");
        return nullptr;
    }

    // Process step
    CollisionResult collisionResult;
    
    step(self->environment, collisionResult);
    
    // Create output dictionary
    PyObject* resultDic = PyDict_New();
    if (resultDic == nullptr) {
        PyErr_NoMemory();
        return nullptr;
    }
    
    vector<int> agentIds;
    collisionResult.getAgentIds(agentIds);

    for(auto itra=agentIds.begin(); itra!=agentIds.end(); ++itra) {
        int agentId = *itra;
        
        vector<int> collisionIds;
        collisionResult.getCollisionIds(agentId, collisionIds);
        
        PyObject* collisionIdTuple = PyTuple_New(collisionIds.size());
        
        int i=0;
        for(auto itrc=collisionIds.begin(); itrc!=collisionIds.end(); ++itrc) {
            int collisionId = *itrc;
            PyObject* item = PyLong_FromLong(collisionId);
            PyTuple_SetItem(collisionIdTuple, i, item);
            // No need to decrease item's refcount.
            i += 1;
        }
        
        PyObject* key = PyLong_FromLong(agentId);
        PyDict_SetItem(resultDic, key, collisionIdTuple);
        
        Py_DECREF(key);
        Py_DECREF(collisionIdTuple);
    }

    return resultDic;
}

static PyObject* Env_render(EnvObject* self, PyObject* args, PyObject* kwds) {
    int cameraId;
    PyObject* posObj = nullptr;
    PyObject* rotObj = nullptr;
    PyObject* ignoreIdsObj = nullptr;

    // Get argument
    const char* kwlist[] = {"camera_id", "pos", "rot", "ignore_ids", nullptr};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iO!O!O!", const_cast<char**>(kwlist),
                                     &cameraId,
                                     &PyArray_Type, &posObj,
                                     &PyArray_Type, &rotObj,
                                     &PyArray_Type, &ignoreIdsObj)) {
        return nullptr;
    }

    if (self->environment == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "rodentia environment not setup");
        return nullptr;
    }
    
    // pos
    const float* posArr = getFloatArrayData(posObj, 3, "pos");
    if( posArr == nullptr ) {
        return nullptr;
    }
    
    Vector3f pos(posArr[0], posArr[1], posArr[2]);

    // rot
    const float* rotArr = getFloatArrayData(rotObj, 4, "rot");
    if( rotArr == nullptr ) {
        return nullptr;
    }

    Quat4f rot(rotArr[0], rotArr[1], rotArr[2], rotArr[3]);

    // ignore_ids
    PyArrayObject* ignoreIdsArray = (PyArrayObject*)ignoreIdsObj;
    if (PyArray_NDIM(ignoreIdsArray) != 1 ) {
        PyErr_Format(PyExc_ValueError, "%s must have shape (%d)", "ignore_ids", 1);
        return nullptr;
    }
    int ignoreIdsLength = PyArray_DIM(ignoreIdsArray, 0);
    const int* ignoreIds_ = getIntArrayData(ignoreIdsObj, ignoreIdsLength, "ignore_ids");
    set<int> ignoreIds;
    for(int i=0; i<ignoreIdsLength; ++i) {
        ignoreIds.insert(ignoreIds_[i]);
    }

    // do render
    render(self->environment, cameraId, pos, rot, ignoreIds);
    
    int frameBufferWidth  = self->environment->getFrameBufferWidth(cameraId);
    int frameBufferHeight = self->environment->getFrameBufferHeight(cameraId);
    const void* frameBuffer = self->environment->getFrameBuffer(cameraId);

    // Create output dictionary
    PyObject* resultDic = PyDict_New();
    if (resultDic == nullptr) {
        PyErr_NoMemory();
        return nullptr;
    }
    
    // Create screen output array
    long* screenDims = new long[3];
    screenDims[0] = frameBufferWidth;
    screenDims[1] = frameBufferHeight;
    screenDims[2] = 3;

    PyArrayObject* screenArray = (PyArrayObject*)PyArray_SimpleNew(
        3, // int nd
        screenDims, // screenDims
        NPY_UINT8); // int typenum
    delete [] screenDims;

    // Set buffer memory to array
    memcpy(PyArray_BYTES(screenArray), frameBuffer, PyArray_NBYTES(screenArray));

    // Put list to dictionary
    PyDict_SetItemString(resultDic, "screen", (PyObject*)screenArray);

    // Decrease ref count of array
    Py_DECREF((PyObject*)screenArray);

    return resultDic;
}

static PyObject* Env_add_box(EnvObject* self, PyObject* args, PyObject* kwds) {
    const char* texturePath = "";
    PyObject* colorObj = nullptr;    
    PyObject* halfExtentObj = nullptr;
    PyObject* posObj = nullptr;
    PyObject* rotObj = nullptr;    
    float mass;
    int detectCollision;
    int visible;

    // Get argument
    const char* kwlist[] = {"texture_path", "color",
                            "half_extent", "pos", "rot", "mass",
                            "detect_collision", "visible",
                            nullptr};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO!O!O!O!fii", const_cast<char**>(kwlist),
                                     &texturePath,
                                     &PyArray_Type, &colorObj,
                                     &PyArray_Type, &halfExtentObj,
                                     &PyArray_Type, &posObj,
                                     &PyArray_Type, &rotObj,
                                     &mass,
                                     &detectCollision,
                                     &visible)) {
        return nullptr;
    }
    
    if (self->environment == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "rodentia environment not setup");
        return nullptr;
    }

    // color
    const float* colorArr = getFloatArrayData(colorObj, 3, "color");
    if( colorArr == nullptr ) {
        return nullptr;
    }

    Vector3f color(colorArr[0], colorArr[1], colorArr[2]);

    // half_extent
    const float* halfExtentArr = getFloatArrayData(halfExtentObj, 3, "half_extent");
    if( halfExtentArr == nullptr ) {
        return nullptr;
    }

    Vector3f halfExtent(halfExtentArr[0], halfExtentArr[1], halfExtentArr[2]);

    // pos
    const float* posArr = getFloatArrayData(posObj, 3, "pos");
    if( posArr == nullptr ) {
        return nullptr;
    }

    Vector3f pos(posArr[0], posArr[1], posArr[2]);

    // rot
    const float* rotArr = getFloatArrayData(rotObj, 4, "rot");
    if( rotArr == nullptr ) {
        return nullptr;
    }
    
    Quat4f rot(rotArr[0], rotArr[1], rotArr[2], rotArr[3]);
    
    int id = addBox(self->environment,
                    texturePath,
                    color,
                    halfExtent, pos, rot,
                    mass,
                    detectCollision != 0,                    
                    visible != 0);
    
    // Returning object ID
    PyObject* idObj = PyLong_FromLong(id);

    return idObj;
}

static PyObject* Env_add_sphere(EnvObject* self, PyObject* args, PyObject* kwds) {
    const char* texturePath = "";
    PyObject* colorObj = nullptr;
    float radius;
    PyObject* posObj = nullptr;
    PyObject* rotObj = nullptr;
    float mass;
    int detectCollision;
    int visible;

    // Get argument
    const char* kwlist[] = {"texture_path", "color",
                            "radius", "pos", "rot", "mass",
                            "detect_collision", "visible",
                            nullptr};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO!fO!O!fii", const_cast<char**>(kwlist),
                                     &texturePath,
                                     &PyArray_Type, &colorObj,
                                     &radius,
                                     &PyArray_Type, &posObj,
                                     &PyArray_Type, &rotObj,
                                     &mass,
                                     &detectCollision,
                                     &visible)) {
        return nullptr;
    }
    
    if (self->environment == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "rodentia environment not setup");
        return nullptr;
    }

    // pos
    const float* posArr = getFloatArrayData(posObj, 3, "pos");
    if( posArr == nullptr ) {
        return nullptr;
    }

    Vector3f pos(posArr[0], posArr[1], posArr[2]);    
    
    // rot
    const float* rotArr = getFloatArrayData(rotObj, 4, "rot");
    if( rotArr == nullptr ) {
        return nullptr;
    }

    Quat4f rot(rotArr[0], rotArr[1], rotArr[2], rotArr[3]);

    // color
    const float* colorArr = getFloatArrayData(colorObj, 3, "color");
    if( colorArr == nullptr ) {
        return nullptr;
    }

    Vector3f color(colorArr[0], colorArr[1], colorArr[2]);
    
    int id = addSphere(self->environment,
                       texturePath,
                       color,
                       radius,
                       pos, rot,
                       mass,
                       detectCollision != 0,
                       visible != 0);

    // Returning object ID
    PyObject* idObj = PyLong_FromLong(id);
    return idObj;
}

static PyObject* Env_add_model(EnvObject* self, PyObject* args, PyObject* kwds) {
    const char* path = "";
    PyObject* colorObj = nullptr;    
    PyObject* scaleObj = nullptr;
    PyObject* posObj = nullptr;
    PyObject* rotObj = nullptr;
    float mass;
    int detectCollision;
    int useMeshCollision;
    int useCollisionFile;
    int visible;

    // Get argument
    const char* kwlist[] = {"path", "color", "scale", "pos", "rot", "mass",
                            "detect_collision", "use_mesh_collision", "use_collision_file",
                            "visible",
                            nullptr};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO!O!O!O!fiiii", const_cast<char**>(kwlist),
                                     &path,
                                     &PyArray_Type, &colorObj,
                                     &PyArray_Type, &scaleObj,
                                     &PyArray_Type, &posObj,
                                     &PyArray_Type, &rotObj,
                                     &mass,
                                     &detectCollision,
                                     &useMeshCollision,
                                     &useCollisionFile,
                                     &visible)) {
        return nullptr;
    }
    
    if (self->environment == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "rodentia environment not setup");
        return nullptr;
    }

    // color
    const float* colorArr = getFloatArrayData(colorObj, 3, "color");
    if( colorArr == nullptr ) {
        return nullptr;
    }

    Vector3f color(colorArr[0], colorArr[1], colorArr[2]);

    // scale
    const float* scaleArr = getFloatArrayData(scaleObj, 3, "scale");
    if( scaleArr == nullptr ) {
        return nullptr;
    }
    
    Vector3f scale(scaleArr[0], scaleArr[1], scaleArr[2]);
    
    // pos
    const float* posArr = getFloatArrayData(posObj, 3, "pos");
    if( posArr == nullptr ) {
        return nullptr;
    }

    Vector3f pos(posArr[0], posArr[1], posArr[2]);
    
    // rot
    const float* rotArr = getFloatArrayData(rotObj, 4, "rot");
    if( rotArr == nullptr ) {
        return nullptr;
    }

    Quat4f rot(rotArr[0], rotArr[1], rotArr[2], rotArr[3]);
    
    int id = addModel(self->environment,
                      path,
                      color,
                      scale, pos, rot,
                      mass,
                      detectCollision != 0,
                      useMeshCollision != 0,
                      useCollisionFile != 0,
                      visible != 0);

    // Returning object ID
    PyObject* idObj = PyLong_FromLong(id);
    return idObj;
}

static PyObject* Env_remove_obj(EnvObject* self, PyObject* args, PyObject* kwds) {
    int id;

    // Get argument
    const char* kwlist[] = {"id", nullptr};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i", const_cast<char**>(kwlist),
                                     &id)) {
        return nullptr;
    }
    
    if (self->environment == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "rodentia environment not setup");
        return nullptr;
    }

    removeObj(self->environment, id);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* Env_locate_object(EnvObject* self, PyObject* args, PyObject* kwds) {
    int id;
    PyObject* posObj = nullptr;
    PyObject* rotObj = nullptr;

    // Get argument
    const char* kwlist[] = {"id", "pos", "rot", nullptr};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iO!O!", const_cast<char**>(kwlist),
                                     &id,
                                     &PyArray_Type, &posObj,
                                     &PyArray_Type, &rotObj)) {
        return nullptr;
    }
    
    if (self->environment == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "rodentia environment not setup");
        return nullptr;
    }

    // pos
    const float* posArr = getFloatArrayData(posObj, 3, "pos");
    if( posArr == nullptr ) {
        return nullptr;
    }
    
    Vector3f pos(posArr[0], posArr[1], posArr[2]);
    
    // rot
    const float* rotArr = getFloatArrayData(rotObj, 4, "rot");
    if( rotArr == nullptr ) {
        return nullptr;
    }

    Quat4f rot(rotArr[0], rotArr[1], rotArr[2], rotArr[3]);
    
    locateObject(self->environment,
                 id,
                 pos, rot);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* Env_locate_agent(EnvObject* self, PyObject* args, PyObject* kwds) {
    int id;
    PyObject* posObj = nullptr;
    float rotY;

    // Get argument
    const char* kwlist[] = {"id", "pos", "rot_y", nullptr};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iO!f", const_cast<char**>(kwlist),
                                     &id,
                                     &PyArray_Type, &posObj,
                                     &rotY)) {
        return nullptr;
    }
    
    if (self->environment == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "rodentia environment not setup");
        return nullptr;
    }

    // pos
    const float* posArr = getFloatArrayData(posObj, 3, "pos");
    if( posArr == nullptr ) {
        return nullptr;
    }

    Vector3f pos(posArr[0], posArr[1], posArr[2]);
    
    locateAgent(self->environment,
                id, pos, rotY);
    
    Py_INCREF(Py_None);
    return Py_None;
}

/**
 * Get result dict object for get_obj_info().
 */
static PyObject* get_info_dic_obj(const EnvironmentObjectInfo& info) {
    // Create output dictionary
    PyObject* resultDic = PyDict_New();
    if (resultDic == nullptr) {
        PyErr_NoMemory();
        return nullptr;
    }

    long* vecDims = new long[1];
    vecDims[0] = 3;

    long* rotDims = new long[1];
    rotDims[0] = 4;

    PyArrayObject* posArray = (PyArrayObject*)PyArray_SimpleNew(
        1, // int nd
        vecDims, // vecDims
        NPY_FLOAT32); // float typenum

    PyArrayObject* velocityArray = (PyArrayObject*)PyArray_SimpleNew(
        1, // int nd
        vecDims, // vecDims
        NPY_FLOAT32); // float typenum

    PyArrayObject* rotArray = (PyArrayObject*)PyArray_SimpleNew(
        1, // int nd
        rotDims, // rotDims
        NPY_FLOAT32); // float typenum

    delete [] vecDims;
    delete [] rotDims;

    memcpy(PyArray_BYTES(posArray), info.pos.getPointer(),
           PyArray_NBYTES(posArray));
    memcpy(PyArray_BYTES(velocityArray), info.velocity.getPointer(),
           PyArray_NBYTES(velocityArray));
    memcpy(PyArray_BYTES(rotArray), info.rot.getPointer(),
           PyArray_NBYTES(rotArray));

    // Put list to dictionary
    PyDict_SetItemString(resultDic, "pos", (PyObject*)posArray);
    PyDict_SetItemString(resultDic, "velocity", (PyObject*)velocityArray);
    PyDict_SetItemString(resultDic, "rot", (PyObject*)rotArray);

    // Decrease ref count of array
    Py_DECREF((PyObject*)posArray);
    Py_DECREF((PyObject*)velocityArray);
    Py_DECREF((PyObject*)rotArray);

    return resultDic;
}

static PyObject* Env_get_obj_info(EnvObject* self, PyObject* args, PyObject* kwds) {
    int id;

    // Get argument
    const char* kwlist[] = {"id", nullptr};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i", const_cast<char**>(kwlist),
                                     &id)) {
        return nullptr;
    }
    
    if (self->environment == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "rodentia environment not setup");
        return nullptr;
    }

    EnvironmentObjectInfo info;
    bool ret = getObjectInfo(self->environment, id, info);

    if(!ret) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    return get_info_dic_obj(info);
}

static PyObject* Env_set_light(EnvObject* self, PyObject* args, PyObject* kwds) {
    PyObject* dirObj = nullptr;
    PyObject* colorObj = nullptr;
    PyObject* ambientColorObj = nullptr;
    float shadowRate;

    // Get argument
    const char* kwlist[] = {"dir", "color", "ambient_color", "shadow_rate", nullptr};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!O!f", const_cast<char**>(kwlist),
                                     &PyArray_Type, &dirObj,
                                     &PyArray_Type, &colorObj,
                                     &PyArray_Type, &ambientColorObj,
                                     &shadowRate) ) {
        return nullptr;
    }
    
    if (self->environment == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "rodentia environment not setup");
        return nullptr;
    }

    // dir
    const float* dirArr = getFloatArrayData(dirObj, 3, "dir");
    if( dirArr == nullptr ) {
        return nullptr;
    }
    const float* colorArr = getFloatArrayData(colorObj, 3, "color");
    if( colorArr == nullptr ) {
        return nullptr;
    }
    const float* ambientColorArr = getFloatArrayData(ambientColorObj, 3, "ambient_color");
    if( ambientColorArr == nullptr ) {
        return nullptr;
    }

    Vector3f dir(dirArr[0], dirArr[1], dirArr[2]);
    Vector3f color(colorArr[0], colorArr[1], colorArr[2]);
    Vector3f ambientColor(ambientColorArr[0], ambientColorArr[1], ambientColorArr[2]);        
    
    setLight(self->environment,
             dir, color, ambientColor,
             shadowRate);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* Env_replace_obj_texture(EnvObject* self, PyObject* args, PyObject* kwds) {
    const char *kwlist[] = { "id",
                             "texture_path",
                             nullptr };
    
    // Get argument
    int id;
    PyObject* texturePathListObj = nullptr;
    
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iO!", const_cast<char**>(kwlist),
                                     &id,
                                     &PyList_Type, &texturePathListObj) ) {
        return nullptr;
    }

    if (self->environment == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "rodentia environment not setup");
        return nullptr;
    }

    vector<string> texturePathes;

    int textureSize = (int)PyList_Size(texturePathListObj);
    for(Py_ssize_t i=0; i<textureSize; ++i) {
        PyObject* textuerPathObj = PyList_GetItem(texturePathListObj, i);

        if (PyUnicode_Check(textuerPathObj)) {
            PyObject * tmpBytes = PyUnicode_AsEncodedString(textuerPathObj,
                                                            "ASCII",
                                                            "strict");
            if (tmpBytes != NULL) {
                const char* textuePathStr = PyBytes_AS_STRING(tmpBytes);
                texturePathes.push_back(textuePathStr);
                Py_DECREF(tmpBytes);
            } else {
                PyErr_Format(PyExc_ValueError, "Replacing texture path was not valid string");
                return nullptr;
            }
        } else {
            PyErr_Format(PyExc_ValueError, "Replacing texture path was not valid string");
            return nullptr;
        }
    }

    replaceObjectTextures(self->environment, id, texturePathes);
    
    Py_INCREF(Py_None);
    return Py_None;
}

// int add_camera_view(width, height, bg_color, near, far, focal_length, shadow_buffer_width)
// int add_agent(radius, pos, rot_y, mass, detect_collision, color)
// void control(id, action)
// void applyImpulse(id, impulse)
// dic step()
// dic render(camera_id, pos, rot)
// int add_box(half_extent, pos, rot, detect_collision)
// int add_sphere(radius, pos, rot, detect_collision)
// int add_model(path, scale, pos, rot, detect_collision)
// void remove_obj(id)
// void locate_object(id, pos, rot)
// void locate_agent(id, pos, rot_y)
// dic get_object_info(id)
// void set_light(dir, color, ambient_color, shadow_rate)
// void replace_obj_texture(id, string[])
// void release()

static PyMethodDef EnvObject_methods[] = {
    {"add_camera_view", (PyCFunction)Env_add_camera_view, METH_VARARGS | METH_KEYWORDS,
     "Add camera view"},
    {"add_agent", (PyCFunction)Env_add_agent, METH_VARARGS | METH_KEYWORDS,
     "Add agent"},
    {"control", (PyCFunction)Env_control, METH_VARARGS | METH_KEYWORDS,
     "Control agent"},
    {"apply_impulse", (PyCFunction)Env_apply_impulse, METH_VARARGS | METH_KEYWORDS,
     "Apply impulse"},
    {"step", (PyCFunction)Env_step, METH_VARARGS | METH_KEYWORDS,
     "Advance the environment"},
    {"render", (PyCFunction)Env_render, METH_VARARGS | METH_KEYWORDS,
     "render screen"},
    {"add_box", (PyCFunction)Env_add_box, METH_VARARGS | METH_KEYWORDS,
     "Add box object"},
    {"add_sphere", (PyCFunction)Env_add_sphere, METH_VARARGS | METH_KEYWORDS,
     "Add sphere object"},
    {"add_model", (PyCFunction)Env_add_model, METH_VARARGS | METH_KEYWORDS,
     "Add model object"},
    {"remove_obj", (PyCFunction)Env_remove_obj, METH_VARARGS | METH_KEYWORDS,
     "Remove object"},
    {"locate_object", (PyCFunction)Env_locate_object, METH_VARARGS | METH_KEYWORDS,
     "Locate object"},
    {"locate_agent", (PyCFunction)Env_locate_agent, METH_VARARGS | METH_KEYWORDS,
     "Locate agent"},
    {"get_obj_info", (PyCFunction)Env_get_obj_info, METH_VARARGS | METH_KEYWORDS,
     "Get object information"},
    {"set_light", (PyCFunction)Env_set_light, METH_VARARGS | METH_KEYWORDS,
     "Set light parameters"},
    {"replace_obj_texture", (PyCFunction)Env_replace_obj_texture, METH_VARARGS | METH_KEYWORDS,
     "Replace object textures"},
    {"release", (PyCFunction)Env_release, METH_VARARGS | METH_KEYWORDS,
     "Release environment"},
    {nullptr}
};


static PyTypeObject rodentia_EnvType = {
    PyVarObject_HEAD_INIT(nullptr, 0) // ob_size
    "rodentia_module.Env",            // tp_name
    sizeof(EnvObject),             // tp_basicsize
    0,                             // tp_itemsize
    (destructor)EnvObject_dealloc, // tp_dealloc
    0,                             // tp_print
    0,                             // tp_getattr
    0,                             // tp_setattr
    0,                             // tp_compare
    0,                             // tp_repr
    0,                             // tp_as_number
    0,                             // tp_as_sequence
    0,                             // tp_as_mapping
    0,                             // tp_hash
    0,                             // tp_call
    0,                             // tp_str
    0,                             // tp_getattro
    0,                             // tp_setattro
    0,                             // tp_as_buffer
    Py_TPFLAGS_DEFAULT,            // tp_flags
    "Env object",                  // tp_doc
    0,                             // tp_traverse
    0,                             // tp_clear
    0,                             // tp_richcompare
    0,                             // tp_weaklistoffset
    0,                             // tp_iter
    0,                             // tp_iternext
    EnvObject_methods,             // tp_methods
    0,                             // tp_members
    0,                             // tp_getset
    0,                             // tp_base
    0,                             // tp_dict
    0,                             // tp_descr_get
    0,                             // tp_descr_set
    0,                             // tp_dictoffset
    (initproc)Env_init,            // tp_init
    0,                             // tp_alloc
    EnvObject_new,                 // tp_new
};



static PyObject* moduleVersion(PyObject* self) {
    return Py_BuildValue("s", RODENTIA_MODULE_VERSION);
}

static PyMethodDef moduleMethods[] = {
    {"version", (PyCFunction)moduleVersion, METH_NOARGS,
     "Module version number."},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef moduleDef = {
    PyModuleDef_HEAD_INIT, "rodentia_module", // m_name
    "3D reinforcement learning environment", // m_doc
    -1,            // m_size
    moduleMethods, // m_methods
    NULL,          // m_reload
    NULL,          // m_traverse
    NULL,          // m_clear
    NULL,          // m_free
};

#ifdef __cplusplus
extern "C" {
#endif

PyMODINIT_FUNC PyInit_rodentia_module() {
    PyObject* m;
    m = PyModule_Create(&moduleDef);

    if (m == NULL) {
        return m;
    }
    
    if (PyType_Ready(&rodentia_EnvType) < 0) {
        return m;
    }
    
    Py_INCREF(&rodentia_EnvType);
    PyModule_AddObject(m, "Env", (PyObject*)&rodentia_EnvType);

    import_array();

    return m;
}

#ifdef __cplusplus
}  // extern "C"
#endif
