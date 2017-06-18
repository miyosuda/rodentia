#ifdef __APPLE__
#include <Python/Python.h>
#else
#include <Python.h>
#endif

//#include <numpy/arrayobject.h>

static PyMethodDef SpamMethods[] = {
	{NULL, NULL, 0, NULL} /* Sentinel */
};

PyMODINIT_FUNC initpyrodent(void) {
	PyObject* m;
	if (m == NULL) {
		return;
	}
}

