#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "Environment.h"

#define RODENT_WRAPPER_VERSION "0.1"

//---------------------------------------------------------
//                    [Interface]
//---------------------------------------------------------

static Environment* createEnvironment() {
	Environment* environment = new Environment();
	return environment;
}

static bool initEnvironment(Environment* environment, int width, int height) {
	environment->init();

	if( !environment->initRenderer(width, height, true) ) {
		return false;
	}
	
	return true;
}

static void releaseEnvironment(Environment* environment) {
	environment->release();
	delete environment;
}

static void stepEnvironment(Environment* environment, const Action& action) {
	environment->step(action);
}

static int getActionSize(Environment* environment) {
	return Action::getActionSize();
}

//---------------------------------------------------------
//                     [Python funcs]
//---------------------------------------------------------

typedef struct {
	PyObject_HEAD
	Environment* environment;
} EnvObject;

static void EnvObject_dealloc(EnvObject* self) {
	if( self->environment != nullptr ) {
		releaseEnvironment(self->environment);
	}
	
	self->ob_type->tp_free((PyObject*)self);
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
	const char* kwlist[] = {"action", nullptr};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", const_cast<char**>(kwlist),
									 &PyArray_Type, &actionObj)) {
		return nullptr;
	}

	if (self->environment == nullptr) {
		PyErr_SetString(PyExc_RuntimeError, "rodent environment not setup");
		return nullptr;
	}

	// Check input action size
	int actionSize = getActionSize(self->environment);

	PyArrayObject* actionArray = (PyArrayObject*)actionObj;

	if (PyArray_NDIM(actionArray) != 1 ||
		PyArray_DIM(actionArray, 0) != actionSize) {
		PyErr_Format(PyExc_ValueError, "action_array must have shape (%i)", actionSize);
		return nullptr;
	}

	if (PyArray_TYPE(actionArray) != NPY_INT) {
		PyErr_SetString(PyExc_ValueError, "action_angle must have dtype np.int32");
		return nullptr;
	}
	
	// Process step
	int* actionArr = (int*)PyArray_DATA(actionArray);
	Action action(actionArr[0], actionArr[1], actionArr[2]);
	
	stepEnvironment(self->environment, action);

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
	screenDims[2] = 4;

	PyArrayObject* screenArray = (PyArrayObject*)PyArray_SimpleNew(
		3, // int nd
		screenDims, // screenDims
		NPY_UINT8); // int typenum
	delete [] screenDims;

	// Set buffer memory to array
	memcpy(PyArray_BYTES(screenArray), frameBuffer, PyArray_NBYTES(screenArray));

	// Put list to dictionary
	PyDict_SetItemString(resultDic, "screen", (PyObject*)screenArray);

	// Put list to dictionary
	//PyDict_SetItemString(resultDic, "reward", PyInt_FromLong(reward)); // TODO: RecRef check

	// Decrease ref count of array
	Py_DECREF((PyObject*)screenArray);
	
	return resultDic;
}

//..
// step(action)
// int add_box(half_extent, pos, rot, detect_collision)
// int add_sphere(radius, pos, rot, detect_collision)
// void remove_obj(id)
// void locate_agent(pos, rot)
//..

static PyMethodDef EnvObject_methods[] = {
	{"step", (PyCFunction)Env_step, METH_VARARGS | METH_KEYWORDS,
	 "Advance the environment"},
	{nullptr}
};


static PyTypeObject rodent_EnvType = {
	PyObject_HEAD_INIT(nullptr) 0, // ob_size
	"rodent.Env",                  // tp_name
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
	return Py_BuildValue("s", RODENT_WRAPPER_VERSION);
}

static PyMethodDef moduleMethods[] = {
	{"version", (PyCFunction)moduleVersion, METH_NOARGS,
	 "Module version number."},
	{nullptr, nullptr, 0, nullptr}
};

#ifdef __cplusplus
extern "C" {
#endif

void initrodent() {
	PyObject* m;

	if (PyType_Ready(&rodent_EnvType) < 0) {
		return;
	}
	
	m = Py_InitModule3("rodent", moduleMethods, "Rodent API module");

	Py_INCREF(&rodent_EnvType);
	PyModule_AddObject(m, "Env", (PyObject*)&rodent_EnvType);

	import_array();
}

#ifdef __cplusplus
}  // extern "C"
#endif
