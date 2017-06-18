#ifdef __APPLE__
#include <Python/Python.h>
#else
#include <Python.h>
#endif

#include <numpy/arrayobject.h>

static PyObject * check(PyObject *self) {
	printf("pass check!!\n");
	Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
	{"check", (PyCFunction)check, METH_NOARGS, "check function.\n"},
	{NULL, NULL, 0, NULL}
};

static char rodent_doc[] = "rodent module\n";

void initrodent() {
	Py_InitModule3("rodent", methods, rodent_doc);
}
