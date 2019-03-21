/*
 * eigen_numpy_list.cpp
 *
 *  Created on: Mar 19, 2019
 *      Author: Gregory Kramida
 *   Copyright: 2019 Gregory Kramida
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */

//standard
#include <vector>

//libraries
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <Eigen/Eigen>
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>

//local
#include "eigen_numpy.hpp"
#include "numpy_conversions_shared.hpp"
#include "../math/tensors.hpp"
#include "../math/typedefs.hpp"

// These macros were renamed in NumPy 1.7.1.
#if !defined(NPY_ARRAY_C_CONTIGUOUS) && defined(NPY_C_CONTIGUOUS)
#define NPY_ARRAY_C_CONTIGUOUS NPY_C_CONTIGUOUS
#endif

#if !defined(NPY_ARRAY_ALIGNED) && defined(NPY_ALIGNED)
#define NPY_ARRAY_ALIGNED NPY_ALIGNED
#endif

namespace bp = boost::python;
namespace np = boost::python::numpy;

using namespace Eigen;

template<class MatType> // MatrixXf or MatrixXd
struct EigenMatrixVectorToPython {
	static PyObject* convert(const std::vector<MatType>& vec) {
		PyObject* py_list = PyList_New(vec.size());
		Py_ssize_t index = 0;
		for (MatType mat : vec) {
			npy_intp shape[2] = { mat.rows(), mat.cols() };
			PyArrayObject* python_array = (PyArrayObject*) PyArray_SimpleNew(
					2, shape, NumpyEquivalentType<typename MatType::Scalar>::type_code);
			copy_array(mat.data(),
					(typename MatType::Scalar*) PyArray_DATA(python_array),
					mat.rows(),
					mat.cols(),
					false,
					true,
					MatType::Flags & Eigen::RowMajorBit);
			PyObject* py_mat = (PyObject*) python_array;
			PyList_SET_ITEM(py_list, index, py_mat);
			index += 1;
		}
		return py_list;
	}
};

template<> // MatrixXf or MatrixXd
struct EigenMatrixVectorToPython<math::MatrixXv2f> {
	static PyObject* convert(const std::vector<math::MatrixXv2f>& vec) {
		PyObject* py_list = PyList_New(vec.size());
		Py_ssize_t index = 0;
		for (math::MatrixXv2f mat : vec) {
			npy_intp shape[3] = { mat.rows(), mat.cols(), 2 };
			PyArrayObject* python_array = (PyArrayObject*) PyArray_SimpleNew(
					3, shape, NumpyEquivalentType<math::MatrixXv2f::Scalar>::type_code);
			copy_array(mat.data(),
					(math::MatrixXv2f::Scalar*) PyArray_DATA(python_array),
					mat.rows(),
					mat.cols(),
					false,
					true,
					math::MatrixXv2f::Flags & Eigen::RowMajorBit);
			PyObject* py_mat = (PyObject*) python_array;
			PyList_SET_ITEM(py_list, index, py_mat);
			index += 1;
		}
		return py_list;
	}
};

//template<class MatType> // MatrixXf or MatrixXd
//struct EigenMatrixVectorToPython2 {
//	static PyObject* convert(const std::vector<MatType>& mat) {
//		bp::tuple shape = bp::make_tuple(mat.rows(), mat.cols());
//		np::dtype dtype = np::dtype::get_builtin<float>();
//		np::ndarray numpy_ndarray = np::zeros(shape, dtype);
//		copy_array(mat.data(),
//				(typename MatType::Scalar*) numpy_ndarray.get_data(),
//				mat.rows(),
//				mat.cols(),
//				false,
//				true,
//				MatType::Flags & Eigen::RowMajorBit);
//		return numpy_ndarray.ptr();
//	}
//};

#define EIGEN_MATRIX_LIST_CONVERTER(Type) \
  bp::to_python_converter<std::vector<Type>, EigenMatrixVectorToPython<Type> >();

#if PY_VERSION_HEX >= 0x03000000
void*
#else
void
#endif
setup_Eigen_list_converters() {
	static bool is_setup = false;
	if (is_setup)
		return NUMPY_IMPORT_ARRAY_RETVAL;
	is_setup = true;
	import_array();

	EIGEN_MATRIX_LIST_CONVERTER(Eigen::MatrixXf);
	EIGEN_MATRIX_LIST_CONVERTER(math::MatrixXv2f);
#if PY_VERSION_HEX >= 0x03000000
	return 0;
#endif
}
