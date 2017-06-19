#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "rodent_api.h"

#define RODENT_WRAPPER_VERSION "0.1"

typedef struct {
	PyObject_HEAD
	void* context;
} EnvObject;

static void EnvObject_dealloc(EnvObject* self) {
	printf("EnvObject_dealloc\n"); //..

	rodent_release(self->context);
	
	self->ob_type->tp_free((PyObject*)self);
}

static PyObject* EnvObject_new(PyTypeObject* type,
							   PyObject* args,
                               PyObject* kwds) {
	printf("EnvObject_new\n"); //..
	
	EnvObject* self;
	
	self = (EnvObject*)type->tp_alloc(type, 0);
	
	if (self != NULL) {
		void* context = rodent_create();
		
		if (context == NULL) {
			PyErr_SetString(PyExc_RuntimeError, "Failed to create rodent environment");
			Py_DECREF(self);
			return NULL;
		}
		
		self->context = context;
	}
	
	return (PyObject*)self;
}

static int Env_init(EnvObject* self, PyObject* args, PyObject* kwds) {
	printf("EnvObject_init\n"); //..
	
	if (self->context == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "rodent environment not setup");
		return -1;
	}

	if (rodent_init(self->context) != 0) {
		PyErr_Format(PyExc_RuntimeError, "Failed to init environment.");
		return -1;
	}

	return 0;
}

static PyObject* Env_step(EnvObject* self, PyObject* args, PyObject* kwds) {
	PyObject* joint_angles_obj = NULL;

	char* kwlist[] = {"joint_angles", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist, &PyArray_Type,
									 &joint_angles_obj)) {
		return NULL;
	}

	int joint_size;
	if( rodent_joint_size(self->context, &joint_size ) != 0 ) {
		PyErr_SetString(PyExc_ValueError, "Failed to get joint size");
		return NULL;
	}

	PyArrayObject* joint_angles_array = (PyArrayObject*)joint_angles_obj;

	if (PyArray_NDIM(joint_angles_array) != 1 ||
		PyArray_DIM(joint_angles_array, 0) != joint_size) {
		PyErr_Format(PyExc_ValueError, "joint_array must have shape (%i)",
					 joint_size);
		return NULL;
	}

	if (PyArray_TYPE(joint_angles_array) != NPY_FLOAT) {
		PyErr_SetString(PyExc_ValueError, "joint_angle must have dtype np.float32");
		return NULL;
	}

	float* data = (float*)PyArray_DATA(joint_angles_array);
	
	if( rodent_step(self->context, data) != 0 ) {
		PyErr_SetString(PyExc_ValueError, "Failed to process step");
		return NULL;
	}

	int itemSize = 3;
	PyObject* list = PyTuple_New(itemSize);
	for (int i=0; i<itemSize; ++i) {
		// TODO:
		PyObject* item = PyFloat_FromDouble((double)i*100);
		PyTuple_SetItem(list, i, item);
	}
	
	return list;
}

static PyMethodDef EnvObject_methods[] = {
	{"step", (PyCFunction)Env_step, METH_VARARGS | METH_KEYWORDS,
	 "Advance the environment"},
	{NULL}
};


static PyTypeObject rodent_EnvType = {
	PyObject_HEAD_INIT(NULL) 0,    /* ob_size */
	"rodent.Env",                  /* tp_name */
	sizeof(EnvObject),             /* tp_basicsize */
	0,                             /* tp_itemsize */
	(destructor)EnvObject_dealloc, /* tp_dealloc */
	0,                             /* tp_print */
	0,                             /* tp_getattr */
	0,                             /* tp_setattr */
	0,                             /* tp_compare */
	0,                             /* tp_repr */
	0,                             /* tp_as_number */
	0,                             /* tp_as_sequence */
	0,                             /* tp_as_mapping */
	0,                             /* tp_hash */
	0,                             /* tp_call */
	0,                             /* tp_str */
	0,                             /* tp_getattro */
	0,                             /* tp_setattro */
	0,                             /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT,            /* tp_flags */
	"Env object",                  /* tp_doc */
	0,                             /* tp_traverse */
	0,                             /* tp_clear */
	0,                             /* tp_richcompare */
	0,                             /* tp_weaklistoffset */
	0,                             /* tp_iter */
	0,                             /* tp_iternext */
	EnvObject_methods,             /* tp_methods */
	0,                             /* tp_members */
	0,                             /* tp_getset */
	0,                             /* tp_base */
	0,                             /* tp_dict */
	0,                             /* tp_descr_get */
	0,                             /* tp_descr_set */
	0,                             /* tp_dictoffset */
	(initproc)Env_init,            /* tp_init */
	0,                             /* tp_alloc */
	EnvObject_new,                 /* tp_new */
};



static PyObject* module_version(PyObject* self) {
	return Py_BuildValue("s", RODENT_WRAPPER_VERSION);
}

static PyMethodDef module_methods[] = {
	{"version", (PyCFunction)module_version, METH_NOARGS,
	 "Module version number."},
	{NULL, NULL, 0, NULL}
};

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
