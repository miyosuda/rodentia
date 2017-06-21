#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "Environment.h"

#define RODENT_WRAPPER_VERSION "0.1"

//..
static Environment* createEnvironment() {
	Environment* environment = new Environment();
	return environment;
}

static bool initEnvironment(Environment* environment) {
	environment->init();

	if( !environment->initRenderer(240, 240, true) ) {
		return false;
	}
	
	return true;
}

static void releaseEnvironment(Environment* environment) {
	environment->release();
	delete environment;
}

static void stepEnvironment(Environment* environment, const float* jointTargetAngles) {
	// TODO: set joint target angles
	environment->step();
}

static int getJointSize(Environment* environment) {
	return 8;
}
//..


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
	// TODO: バッファのサイズを引数で取る様に
	
	if (self->environment == nullptr) {
		PyErr_SetString(PyExc_RuntimeError, "rodent environment not setup");
		return -1;
	}

	if ( !initEnvironment(self->environment) ) {
		PyErr_Format(PyExc_RuntimeError, "Failed to init environment.");
		return -1;
	}

	return 0;
}

static PyObject* Env_step(EnvObject* self, PyObject* args, PyObject* kwds) {
	PyObject* joint_angles_obj = nullptr;

	const char* kwlist[] = {"joint_angles", nullptr};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", const_cast<char**>(kwlist), &PyArray_Type,
									 &joint_angles_obj)) {
		return nullptr;
	}

	if (self->environment == nullptr) {
		PyErr_SetString(PyExc_RuntimeError, "rodent environment not setup");
		return nullptr;
	}

	// Check input joint size
	int joint_size = getJointSize(self->environment);

	PyArrayObject* joint_angles_array = (PyArrayObject*)joint_angles_obj;

	if (PyArray_NDIM(joint_angles_array) != 1 ||
		PyArray_DIM(joint_angles_array, 0) != joint_size) {
		PyErr_Format(PyExc_ValueError, "joint_array must have shape (%i)",
					 joint_size);
		return nullptr;
	}

	if (PyArray_TYPE(joint_angles_array) != NPY_FLOAT) {
		PyErr_SetString(PyExc_ValueError, "joint_angle must have dtype np.float32");
		return nullptr;
	}
	
	// Process step
	float* data = (float*)PyArray_DATA(joint_angles_array);
	stepEnvironment(self->environment, data);

	// Create output dictionary
	PyObject* result = PyDict_New();
	if (result == NULL) {
		PyErr_NoMemory();
		return NULL;
	}	

	// Create output joint tuple
	// TODO: arrayに変更?
	int itemSize = 3;
	PyObject* list = PyTuple_New(itemSize);
	for (int i=0; i<itemSize; ++i) {
		PyObject* item = PyFloat_FromDouble((double)i*100);
		PyTuple_SetItem(list, i, item);
	}

	int frameBufferWidth  = self->environment->getFrameBufferWidth();
	int frameBufferHeight = self->environment->getFrameBufferHeight();
	const void* frameBuffer = self->environment->getFrameBuffer();

	// Create screen outpu array
	long* dims = new long[3];
	dims[0] = frameBufferWidth;
	dims[1] = frameBufferHeight;
	dims[2] = 4;

    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew(
        3, // int nd
		dims, // dims
        NPY_UINT8); // int typenum
    delete [] dims;

	// Set buffer memory to array
    memcpy(PyArray_BYTES(array), frameBuffer, PyArray_NBYTES(array));

	// Put list to dictionary
    PyDict_SetItemString(result, "screen", (PyObject*)array);

	// Put list to dictionary
    PyDict_SetItemString(result, "joint_angles", list);

	// Decrease ref count of array
    Py_DECREF((PyObject*)array);
	
	// Decrease ref count of list
    Py_DECREF(list);
	
	return result;
}

static PyMethodDef EnvObject_methods[] = {
	{"step", (PyCFunction)Env_step, METH_VARARGS | METH_KEYWORDS,
	 "Advance the environment"},
	{NULL}
};


static PyTypeObject rodent_EnvType = {
	PyObject_HEAD_INIT(NULL) 0,    // ob_size
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



static PyObject* module_version(PyObject* self) {
	return Py_BuildValue("s", RODENT_WRAPPER_VERSION);
}

static PyMethodDef module_methods[] = {
	{"version", (PyCFunction)module_version, METH_NOARGS,
	 "Module version number."},
	{NULL, NULL, 0, NULL}
};

#ifdef __cplusplus
extern "C" {
#endif

void initrodent() {
	PyObject* m;

	if (PyType_Ready(&rodent_EnvType) < 0) {
		return;
	}
	
	m = Py_InitModule3("rodent", module_methods, "Rodent API module");

	Py_INCREF(&rodent_EnvType);
	PyModule_AddObject(m, "Env", (PyObject*)&rodent_EnvType);

	import_array();
}

#ifdef __cplusplus
}  // extern "C"
#endif
