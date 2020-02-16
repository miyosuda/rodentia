#include <Python.h>

#include <vector>
#include <string>
using namespace std;

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "Environment.h"
#include "Action.h"
#include "Vector3f.h"
#include "EnvironmentObject.h"

#define RODENTIA_MODULE_VERSION "0.0.1"


//---------------------------------------------------------
//                    [Interface]
//---------------------------------------------------------

static Environment* createEnvironment() {
	Environment* environment = new Environment();
	return environment;
}

static bool initEnvironment(Environment* environment, int width, int height,
							float bgColorR, float bgColorG, float bgColorB) {
	if( !environment->init(width, height, Vector3f(bgColorR, bgColorG, bgColorB)) ) {
		return false;
	}
	
	return true;
}

static void releaseEnvironment(Environment* environment) {
	environment->release();
	delete environment;
}

static void stepEnvironment(Environment* environment, const Action& action, int stepNum) {
	environment->step(action, stepNum, true);
}

static int addBox(Environment* environment,
				  const char* texturePath,
				  float halfExtentX, float halfExtentY, float halfExtentZ,
				  float posX, float posY, float posZ,
				  float rotX, float rotY, float rotZ,
				  float mass,
				  bool detectCollision) {
	return environment->addBox(texturePath,
							   Vector3f(halfExtentX, halfExtentY, halfExtentZ),
							   Vector3f(posX, posY, posZ),
							   Vector3f(rotX, rotY, rotZ),
							   mass,
							   detectCollision);
}

static int addSphere(Environment* environment,
					 const char* texturePath,
					 float radius,
					 float posX, float posY, float posZ,
                     float rotX, float rotY, float rotZ,
					 float mass,
					 bool detectCollision) {
	return environment->addSphere(texturePath,
								  radius,
								  Vector3f(posX, posY, posZ),
                                  Vector3f(rotX, rotY, rotZ),
								  mass,
								  detectCollision);
}

static int addModel(Environment* environment,
					const char* path,
					float scaleX, float scaleY, float scaleZ,
					float posX, float posY, float posZ,
					float rotX, float rotY, float rotZ,
					float mass,
					bool detectCollision) {
	return environment->addModel(path,
								 Vector3f(scaleX, scaleY, scaleZ),
								 Vector3f(posX, posY, posZ),
                                 Vector3f(rotX, rotY, rotZ),
								 mass,
								 detectCollision);
}

static void removeObj(Environment* environment, 
					  int id) {
	environment->removeObject(id);
}

static void locateObject(Environment* environment,
						 int id,
						 float posX, float posY, float posZ,
						 float rotX, float rotY, float rotZ) {
	environment->locateObject(id,
							  Vector3f(posX, posY, posZ),
							  Vector3f(rotX, rotY, rotZ));
}

static void locateAgent(Environment* environment,
						float posX, float posY, float posZ,
						float rot) {
	environment->locateAgent(Vector3f(posX, posY, posZ), rot);
}

static bool getObjectInfo(Environment* environment,
						  int id, EnvironmentObjectInfo& info) {
	return environment->getObjectInfo(id, info);
}

static bool getAgentInfo(Environment* environment,
						 EnvironmentObjectInfo& info) {
	return environment->getAgentInfo(info);
}

static void setLight(Environment* environment,
					 float dirX, float dirY, float dirZ,
					 float colorX, float colorY, float colorZ,
					 float ambientColorX, float ambientColorY, float ambientColorZ,
					 float shadowColorRate) {
	environment->setLight(Vector3f(dirX, dirY, dirZ),
						  Vector3f(colorX, colorY, colorZ),
						  Vector3f(ambientColorX, ambientColorY, ambientColorZ),
						  shadowColorRate);
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
	const char *kwlist[] = { "width",
							 "height",
							 "bg_color",
							 nullptr };

	// Get argument
	int width;
	int height;
	PyObject* bgColorObj = nullptr;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiO!", const_cast<char**>(kwlist),
									 &width,
									 &height,
									 &PyArray_Type, &bgColorObj)) {
		PyErr_SetString(PyExc_RuntimeError, "init argument shortage");
		return -1;
	}

	if (self->environment == nullptr) {
		PyErr_SetString(PyExc_RuntimeError, "rodentia environment not setup");
		return -1;
	}

	const float* bgColorArr = getFloatArrayData(bgColorObj, 3, "bg_color");
	if( bgColorArr == nullptr ) {
		return -1;
	}

	float bgColorR = bgColorArr[0];
	float bgColorG = bgColorArr[1];
	float bgColorB = bgColorArr[2];

	// Initialize environment
	if ( !initEnvironment(self->environment, width, height,
						  bgColorR, bgColorG, bgColorB) ) {
		PyErr_Format(PyExc_RuntimeError, "Failed to init environment.");
		return -1;
	}

	return 0;
}

static PyObject* Env_step(EnvObject* self, PyObject* args, PyObject* kwds) {
	PyObject* actionObj = nullptr;

	// Get argument
	const char* kwlist[] = {"action", "num_steps", nullptr};

	int stepNum = 1;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!i", const_cast<char**>(kwlist),
									 &PyArray_Type, &actionObj,
									 &stepNum)) {
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

	// Process step
	stepEnvironment(self->environment, action, stepNum);

	// Create output dictionary
	PyObject* resultDic = PyDict_New();
	if (resultDic == nullptr) {
		PyErr_NoMemory();
		return nullptr;
	}

	int frameBufferWidth  = self->environment->getFrameBufferWidth();
	int frameBufferHeight = self->environment->getFrameBufferHeight();
	const void* frameBuffer = self->environment->getFrameBuffer();

	// Create screen outpu array
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

	const set<int>& collidedIds = self->environment->getCollidedIds();

	size_t collidedIdSize = collidedIds.size();
	PyObject* collidedIdTuple = PyTuple_New(collidedIdSize);
	
	int i=0;
	for(auto it=collidedIds.begin(); it!=collidedIds.end(); ++it) {
		int collidedId = *it;
		PyObject* item = PyLong_FromLong(collidedId);
		PyTuple_SetItem(collidedIdTuple, i, item);
		// No need to decrease item's refcount.
		i += 1;
	}

	// Put tuple to dictionary
	PyDict_SetItemString(resultDic, "collided", collidedIdTuple);

	// Decrease ref count of array
	Py_DECREF((PyObject*)screenArray);

	// Decrease ref count of tuple
    Py_DECREF(collidedIdTuple);
	
	return resultDic;
}

static PyObject* Env_add_box(EnvObject* self, PyObject* args, PyObject* kwds) {
	const char* texturePath = "";
	PyObject* halfExtentObj = nullptr;
	PyObject* posObj = nullptr;
	PyObject* rotObj = nullptr;
	float mass;
	int detectCollision;

	// Get argument
	const char* kwlist[] = {"texture_path", "half_extent", "pos", "rot", "mass", "detect_collision",
							nullptr};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO!O!O!fi", const_cast<char**>(kwlist),
									 &texturePath,
									 &PyArray_Type, &halfExtentObj,
									 &PyArray_Type, &posObj,
									 &PyArray_Type, &rotObj,
									 &mass,
									 &detectCollision)) {
		return nullptr;
	}
	
	if (self->environment == nullptr) {
		PyErr_SetString(PyExc_RuntimeError, "rodentia environment not setup");
		return nullptr;
	}

	// half_extent
	const float* halfExtentArr = getFloatArrayData(halfExtentObj, 3, "half_extent");
	if( halfExtentArr == nullptr ) {
		return nullptr;
	}
	
	float halfExtentX = halfExtentArr[0];
	float halfExtentY = halfExtentArr[1];
	float halfExtentZ = halfExtentArr[2];

	// pos
	const float* posArr = getFloatArrayData(posObj, 3, "pos");
	if( posArr == nullptr ) {
		return nullptr;
	}
	
	float posX = posArr[0];
	float posY = posArr[1];
	float posZ = posArr[2];

	// rot
	const float* rotArr = getFloatArrayData(rotObj, 3, "rot");
	if( rotArr == nullptr ) {
		return nullptr;
	}
	
	float rotX = rotArr[0];
	float rotY = rotArr[1];
	float rotZ = rotArr[2];
    
	int id = addBox(self->environment,
					texturePath,
					halfExtentX, halfExtentY, halfExtentZ,
					posX, posY, posZ,
					rotX, rotY, rotZ,
					mass,
					detectCollision != 0);
	
	// Returning object ID	
	PyObject* idObj = PyLong_FromLong(id);

	return idObj;
}

static PyObject* Env_add_sphere(EnvObject* self, PyObject* args, PyObject* kwds) {
	const char* texturePath = "";
	float radius;
	PyObject* posObj = nullptr;
	PyObject* rotObj = nullptr;
	float mass;
	int detectCollision;

	// Get argument
	const char* kwlist[] = {"texture_path", "radius", "pos", "rot", "mass", "detect_collision",
							nullptr};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "sfO!O!fi", const_cast<char**>(kwlist),
									 &texturePath,
									 &radius,
									 &PyArray_Type, &posObj,
									 &PyArray_Type, &rotObj,
									 &mass,
									 &detectCollision)) {
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
	
	float posX = posArr[0];
	float posY = posArr[1];
	float posZ = posArr[2];

	// rot
	const float* rotArr = getFloatArrayData(rotObj, 3, "rot");
	if( rotArr == nullptr ) {
		return nullptr;
	}
	
	float rotX = rotArr[0];
	float rotY = rotArr[1];
	float rotZ = rotArr[2];    

	int id = addSphere(self->environment,
					   texturePath,
					   radius,
					   posX, posY, posZ,
					   rotX, rotY, rotZ,
					   mass,
					   detectCollision != 0);

	// Returning object ID
	PyObject* idObj = PyLong_FromLong(id);
	return idObj;
}

static PyObject* Env_add_model(EnvObject* self, PyObject* args, PyObject* kwds) {
	const char* path = "";
	PyObject* scaleObj = nullptr;
	PyObject* posObj = nullptr;
	PyObject* rotObj = nullptr;
	float mass;
	int detectCollision;

	// Get argument
	const char* kwlist[] = {"path", "scale", "pos", "rot", "mass", "detect_collision", nullptr};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO!O!O!fi", const_cast<char**>(kwlist),
									 &path,
									 &PyArray_Type, &scaleObj,
									 &PyArray_Type, &posObj,
									 &PyArray_Type, &rotObj,
									 &mass,
									 &detectCollision)) {
		return nullptr;
	}
	
	if (self->environment == nullptr) {
		PyErr_SetString(PyExc_RuntimeError, "rodentia environment not setup");
		return nullptr;
	}

	// scale
	const float* scaleArr = getFloatArrayData(scaleObj, 3, "scale");
	if( scaleArr == nullptr ) {
		return nullptr;
	}
	
	float scaleX = scaleArr[0];
	float scaleY = scaleArr[1];
	float scaleZ = scaleArr[2];	

	// pos
	const float* posArr = getFloatArrayData(posObj, 3, "pos");
	if( posArr == nullptr ) {
		return nullptr;
	}
	
	float posX = posArr[0];
	float posY = posArr[1];
	float posZ = posArr[2];

	// rot
	const float* rotArr = getFloatArrayData(rotObj, 3, "rot");
	if( rotArr == nullptr ) {
		return nullptr;
	}
	
	float rotX = rotArr[0];
	float rotY = rotArr[1];
	float rotZ = rotArr[2];    

	int id = addModel(self->environment,
					  path,
					  scaleX, scaleY, scaleZ,
					  posX, posY, posZ,
					  rotX, rotY, rotZ,
					  mass,
					  detectCollision != 0);

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
	
	float posX = posArr[0];
	float posY = posArr[1];
	float posZ = posArr[2];

	// rot
	const float* rotArr = getFloatArrayData(rotObj, 3, "rot");
	if( rotArr == nullptr ) {
		return nullptr;
	}
	
	float rotX = rotArr[0];
	float rotY = rotArr[1];
	float rotZ = rotArr[2];    

	locateObject(self->environment,
				 id,
				 posX, posY, posZ,
				 rotX, rotY, rotZ);

	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject* Env_locate_agent(EnvObject* self, PyObject* args, PyObject* kwds) {
	PyObject* posObj = nullptr;
	float rot;

	// Get argument
	const char* kwlist[] = {"pos", "rot", nullptr};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!f", const_cast<char**>(kwlist),
									 &PyArray_Type, &posObj,
									 &rot)) {
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
	
	float posX = posArr[0];
	float posY = posArr[1];
	float posZ = posArr[2];

	locateAgent(self->environment,
				posX, posY, posZ,
				rot);

	Py_INCREF(Py_None);
	return Py_None;
}

/**
 * Common function to get result dict object for get_obj_info() and get_agent_info().
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

	PyArrayObject* posArray = (PyArrayObject*)PyArray_SimpleNew(
		1, // int nd
		vecDims, // vecims
		NPY_FLOAT32); // float typenum

	PyArrayObject* velocityArray = (PyArrayObject*)PyArray_SimpleNew(
		1, // int nd
		vecDims, // vecims
		NPY_FLOAT32); // float typenum

	PyArrayObject* eulerAnglesArray = (PyArrayObject*)PyArray_SimpleNew(
		1, // int nd
		vecDims, // vecims
		NPY_FLOAT32); // float typenum

	delete [] vecDims;

	memcpy(PyArray_BYTES(posArray), info.pos.getPointer(),
		   PyArray_NBYTES(posArray));
	memcpy(PyArray_BYTES(velocityArray), info.velocity.getPointer(),
		   PyArray_NBYTES(velocityArray));
	memcpy(PyArray_BYTES(eulerAnglesArray), info.eulerAngles.getPointer(),
		   PyArray_NBYTES(eulerAnglesArray));

	// Put list to dictionary
	PyDict_SetItemString(resultDic, "pos", (PyObject*)posArray);
	PyDict_SetItemString(resultDic, "velocity", (PyObject*)velocityArray);
	PyDict_SetItemString(resultDic, "euler_angles", (PyObject*)eulerAnglesArray);

	// Decrease ref count of array
	Py_DECREF((PyObject*)posArray);
	Py_DECREF((PyObject*)velocityArray);
	Py_DECREF((PyObject*)eulerAnglesArray);

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

static PyObject* Env_get_agent_info(EnvObject* self, PyObject* args, PyObject* kwds) {
	if (self->environment == nullptr) {
		PyErr_SetString(PyExc_RuntimeError, "rodentia environment not setup");
		return nullptr;
	}

	EnvironmentObjectInfo info;
	bool ret = getAgentInfo(self->environment, info);

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
	
	setLight(self->environment,
			 dirArr[0], dirArr[1], dirArr[2],
			 colorArr[0], colorArr[1], colorArr[2],
			 ambientColorArr[0], ambientColorArr[1], ambientColorArr[2],
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

// dic step(action)
// int add_box(half_extent, pos, rot, detect_collision)
// int add_sphere(radius, pos, rot, detect_collision)
// int add_model(path, scale, pos, rot, detect_collision)
// void remove_obj(id)
// void locate_object(id, pos, rot)
// void locate_agent(pos, rot)
// dic get_object_info(id)
// dic get_agent_info()
// void set_light(dir, color, ambient_color, shadow_rate)
// void replace_obj_texture(id, string[])

static PyMethodDef EnvObject_methods[] = {
	{"step", (PyCFunction)Env_step, METH_VARARGS | METH_KEYWORDS,
	 "Advance the environment"},
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
	{"get_agent_info", (PyCFunction)Env_get_agent_info, METH_VARARGS | METH_KEYWORDS,
	 "Get agent information"},
	{"set_light", (PyCFunction)Env_set_light, METH_VARARGS | METH_KEYWORDS,
	 "Set light parameters"},
	{"replace_obj_texture", (PyCFunction)Env_replace_obj_texture, METH_VARARGS | METH_KEYWORDS,
	 "Replace object textures"},
	{nullptr}
};


static PyTypeObject rodentia_EnvType = {
	PyVarObject_HEAD_INIT(nullptr, 0) // ob_size
	"rodentia_module.Env",           // tp_name
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

PyMODINIT_FUNC
PyInit_rodentia_module()
{
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