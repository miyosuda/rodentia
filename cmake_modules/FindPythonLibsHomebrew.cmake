# - Find python libraries with Homebrew environment
# 
# This module finds the libraries corresponding to the Python interpeter
# FindPythonInterp provides.
# This code sets the following variables:
#
#  PYTHONLIBS_FOUND           - have the Python libs been found
#  PYTHON_LIBRARIES           - path to the python library
#  PYTHON_INCLUDE_DIRS        - path to where Python.h is found

find_package(PythonInterp REQUIRED)

if(NOT PYTHONINTERP_FOUND)
    set(PYTHONLIBS_FOUND FALSE)
    return()
endif()

STRING( REGEX REPLACE "([0-9.]+)\\.[0-9]+" "\\1" PYTHON_SHORT_VERSION ${PYTHON_VERSION_STRING} )

execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
	"from distutils.sysconfig import get_python_inc; print(get_python_inc())"
	RESULT_VARIABLE _PYTHON_LIB_INC_SUCCESS
    OUTPUT_VARIABLE _PYTHON_LIB_INC_VALUE
    ERROR_VARIABLE  _PYTHON_LIB_INC_ERROR_VALUE
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT _PYTHON_LIB_INC_SUCCESS MATCHES 0)
    if(PythonLibsHomebrew_FIND_REQUIRED)
        message(FATAL_ERROR
            "Python library config failure:\n${_PYTHON_LIB_INC_ERROR_VALUE}")
    endif()
    set(PYTHONLIBS_FOUND FALSE)
    return()
endif()

execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
	"import distutils.sysconfig as sysconfig; import os; print(sysconfig.get_config_var('LIBDIR'))"
	RESULT_VARIABLE _PYTHON_LIB_LIB_SUCCESS
    OUTPUT_VARIABLE _PYTHON_LIB_LIB_VALUE
    ERROR_VARIABLE  _PYTHON_LIB_LIB_ERROR_VALUE
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT _PYTHON_LIB_LIB_SUCCESS MATCHES 0)
    if(PythonLibsHomebrew_FIND_REQUIRED)
        message(FATAL_ERROR
            "Python library config failure:\n${_PYTHON_LIB_LIB_ERROR_VALUE}")
    endif()
    set(PYTHONLIBS_FOUND FALSE)
    return()
endif()


set(PYTHON_INCLUDE_DIRS ${_PYTHON_LIB_INC_VALUE})
set(PYTHON_LIBRARIES "${_PYTHON_LIB_LIB_VALUE}/libpython${PYTHON_SHORT_VERSION}.dylib")

set(PYTHONLIBS_FOUND TRUE)
