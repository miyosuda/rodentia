#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "Environment.h"
#include "Action.h"
#include "Vector3f.h"

#define RODENT_MODULE_VERSION "0.0.1"

#if PY_MAJOR_VERSION >= 3
#define PyInt_FromLong PyLong_FromLong
#define PyString_FromString PyBytes_FromString
#endif

//---------------------------------------------------------
//                    [Interface]
//---------------------------------------------------------

static Environment* createEnvironment() {
	Environment* environment = new Environment();
	return environment;
}

static bool initEnvironment(Environment* environment, int width, int height) {
	if( !environment->init(width, height, true) ) {
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
				  float halfExtentX, float halfExtentY, float halfExtentZ,
				  float posX, float posY, float posZ,
				  float rot,
				  bool detectCollision) {
	return environment->addBox(Vector3f(halfExtentX, halfExtentY, halfExtentZ),
							   Vector3f(posX, posY, posZ),
							   rot,
							   detectCollision);
}

static int addSphere(Environment* environment,
					 float radius,
					 float posX, float posY, float posZ,
					 float rot,
					 bool detectCollision) {
	return environment->addSphere(radius,
								  Vector3f(posX, posY, posZ),
								  rot,
								  detectCollision);
}

static int addModel(Environment* environment,
					const char* path,
					float scaleX, float scaleY, float scaleZ,
					float posX, float posY, float posZ,
					float rot,
					bool detectCollision) {
	return environment->addModel(path,
								 Vector3f(scaleX, scaleY, scaleZ),
								 Vector3f(posX, posY, posZ),
								 rot,
								 detectCollision);
}

static void removeObj(Environment* environment, 
					  int id) {
	environment->removeObject(id);
}

static void locateAgent(Environment* environment,
						float posX, float posY, float posZ,
						float rot) {
	environment->locateAgent(Vector3f(posX, posY, posZ), rot);
}

static void setLightDir(Environment* environment,
						float dirX, float dirY, float dirZ) {
	environment->setLightDir(Vector3f(dirX, dirY, dirZ));
}

static int getActionSize(Environment* environment) {
	return Action::getActionSize();
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

#if PY_MAJOR_VERSION >= 3
	(((PyObject*)(self))->ob_type)->tp_free((PyObject*)self);
#else
	self->ob_type->tp_free((PyObject*)self);
#endif
}

static PyObject* EnvObject_new(PyTypeObject* type,
							   PyObject* args,
							   PyObject* kwds) {
	EnvObject* self;
	
	self = (EnvObject*)type->tp_alloc(type, 0);
	
	if (self != nullptr) {
		Environment* environment = createEnvironment();
		if (environment == nullptr) {
			PyErr_SetString(PyExc_RuntimeError, "Failed to create rodent environment");
			Py_DECREF(self);
			return nullptr;
		}
		
		self->environment = environment;
	}
	
	return (PyObject*)self;
}

static int Env_init(EnvObject* self, PyObject* args, PyObject* kwds) {
	const char *kwlist[] = { "width", "height", nullptr };

	// Get argument
	int width;
	int height;
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "ii", const_cast<char**>(kwlist),
									 &width, &height)) {
		PyErr_SetString(PyExc_RuntimeError, "init argument shortage");
		return -1;
	}
	
	if (self->environment == nullptr) {
		PyErr_SetString(PyExc_RuntimeError, "rodent environment not setup");
		return -1;
	}

	// Initialize environment
	if ( !initEnvironment(self->environment, width, height) ) {
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
		PyErr_SetString(PyExc_RuntimeError, "rodent environment not setup");
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
		PyObject* item = PyInt_FromLong(collidedId);
		PyTuple_SetItem(collidedIdTuple, i, item);
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
	PyObject* halfExtentObj = nullptr;
	PyObject* posObj = nullptr;
	float rot;
	int detectCollision;

	// Get argument
	const char* kwlist[] = {"half_extent", "pos", "rot", "detect_collision", nullptr};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!fi", const_cast<char**>(kwlist),
									 &PyArray_Type, &halfExtentObj,
									 &PyArray_Type, &posObj,
									 &rot,
									 &detectCollision)) {
		return nullptr;
	}
	
	if (self->environment == nullptr) {
		PyErr_SetString(PyExc_RuntimeError, "rodent environment not setup");
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

	int id = addBox(self->environment,
					halfExtentX, halfExtentY, halfExtentZ,
					posX, posY, posZ,
					rot, detectCollision != 0);
	
	// Returning object ID	
	PyObject* idObj = PyInt_FromLong(id);

	return idObj;
}

static PyObject* Env_add_sphere(EnvObject* self, PyObject* args, PyObject* kwds) {
	float radius;
	PyObject* posObj = nullptr;
	float rot;
	int detectCollision;

	// Get argument
	const char* kwlist[] = {"radius", "pos", "rot", "detect_collision", nullptr};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "fO!fi", const_cast<char**>(kwlist),
									 &radius,
									 &PyArray_Type, &posObj,
									 &rot,
									 &detectCollision)) {
		return nullptr;
	}
	
	if (self->environment == nullptr) {
		PyErr_SetString(PyExc_RuntimeError, "rodent environment not setup");
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

	int id = addSphere(self->environment,
					   radius,
					   posX, posY, posZ,
					   rot, detectCollision != 0);

	// Returning object ID
	PyObject* idObj = PyInt_FromLong(id);
	return idObj;
}

static PyObject* Env_add_model(EnvObject* self, PyObject* args, PyObject* kwds) {
	const char* path = "";
	PyObject* scaleObj = nullptr;	
	PyObject* posObj = nullptr;
	float rot;
	int detectCollision;

	// Get argument
	const char* kwlist[] = {"path", "scale", "pos", "rot", "detect_collision", nullptr};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO!O!fi", const_cast<char**>(kwlist),
									 &path,
									 &PyArray_Type, &scaleObj,
									 &PyArray_Type, &posObj,
									 &rot,
									 &detectCollision)) {
		return nullptr;
	}
	
	if (self->environment == nullptr) {
		PyErr_SetString(PyExc_RuntimeError, "rodent environment not setup");
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

	int id = addModel(self->environment,
					  path,
					  scaleX, scaleY, scaleZ,
					  posX, posY, posZ,
					  rot, detectCollision != 0);

	// Returning object ID
	PyObject* idObj = PyInt_FromLong(id);
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
		PyErr_SetString(PyExc_RuntimeError, "rodent environment not setup");
		return nullptr;
	}

	removeObj(self->environment, id);

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
		PyErr_SetString(PyExc_RuntimeError, "rodent environment not setup");
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

static PyObject* Env_set_light_dir(EnvObject* self, PyObject* args, PyObject* kwds) {
	PyObject* dirObj = nullptr;

	// Get argument
	const char* kwlist[] = {"dir", nullptr};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", const_cast<char**>(kwlist),
									 &PyArray_Type, &dirObj) ) {
		return nullptr;
	}
	
	if (self->environment == nullptr) {
		PyErr_SetString(PyExc_RuntimeError, "rodent environment not setup");
		return nullptr;
	}

	// dir
	const float* dirArr = getFloatArrayData(dirObj, 3, "dir");
	if( dirArr == nullptr ) {
		return nullptr;
	}
	
	float dirX = dirArr[0];
	float dirY = dirArr[1];
	float dirZ = dirArr[2];

	setLightDir(self->environment,
				dirX, dirY, dirZ);

	Py_INCREF(Py_None);
	return Py_None;
}

// dic step(action)
// int add_box(half_extent, pos, rot, detect_collision)
// int add_sphere(radius, pos, rot, detect_collision)
// int add_model(path, scale, pos, rot, detect_collision)
// void remove_obj(id)
// void locate_agent(pos, rot)

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
	{"locate_agent", (PyCFunction)Env_locate_agent, METH_VARARGS | METH_KEYWORDS,
	 "Locate agent"},
	{"set_light_dir", (PyCFunction)Env_set_light_dir, METH_VARARGS | METH_KEYWORDS,
	 "Set directional light direction"},
	{nullptr}
};


static PyTypeObject rodent_EnvType = {
#if PY_MAJOR_VERSION >= 3
	PyVarObject_HEAD_INIT(nullptr, 0) // ob_size
#else
	PyObject_HEAD_INIT(nullptr) 0, // ob_size	
#endif
	"rodent_module.Env",           // tp_name
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
	return Py_BuildValue("s", RODENT_MODULE_VERSION);
}

static PyMethodDef moduleMethods[] = {
	{"version", (PyCFunction)moduleVersion, METH_NOARGS,
	 "Module version number."},
	{nullptr, nullptr, 0, nullptr}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduleDef = {
	PyModuleDef_HEAD_INIT, "rodent_module", // m_name
	"3D reinforcement learning environment", // m_doc
	-1,            // m_size
	moduleMethods, // m_methods
	NULL,          // m_reload
	NULL,          // m_traverse
	NULL,          // m_clear
	NULL,          // m_free
};
#endif

#ifdef __cplusplus
extern "C" {
#endif

PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit_rodent_module()
#else
initrodent_module()
#endif
{
	PyObject* m;
#if PY_MAJOR_VERSION >= 3
	m = PyModule_Create(&moduleDef);
#else
	m = Py_InitModule3("rodent_module", moduleMethods, "Rodent API module");
#endif

#if PY_MAJOR_VERSION >= 3
	if (m == NULL) {
		return m;
	}
#else
	if (m == NULL) {
		return;
	}
#endif
	
	if (PyType_Ready(&rodent_EnvType) < 0) {
#if PY_MAJOR_VERSION >= 3
		return m;
#else
		return;
#endif
	}
	
	Py_INCREF(&rodent_EnvType);
	PyModule_AddObject(m, "Env", (PyObject*)&rodent_EnvType);

	import_array();

#if PY_MAJOR_VERSION >= 3
	return m;
#endif
}

#ifdef __cplusplus
}  // extern "C"
#endif
