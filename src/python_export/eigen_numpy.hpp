#pragma once

#include <Python.h>

#if PY_VERSION_HEX >= 0x03000000
void* setup_Eigen_matrix_converters();
void* setup_Eigen_tensor_converters();
void* setup_Eigen_list_converters();
#else
void setup_Eigen_matrix_converters();
void setup_Eigen_tensor_converters();
void setup_Eigen_list_converters();
#endif
