#ifndef _EIGEN_NUMPY_H_
#define _EIGEN_NUMPY_H_

#if PY_VERSION_HEX >= 0x03000000
void* setup_Eigen_matrix_converters();
void* setup_Eigen_tensor_converters();
#else
void setup_Eigen_matrix_converters();
void setup_Eigen_tensor_converters();
#endif

#endif
