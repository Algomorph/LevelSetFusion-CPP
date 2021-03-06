/*
 * eigen_numpy_tensor.cpp
 *
 *  Created on: Feb 5, 2019
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

//standard library
#include <array>
#include <cstdlib>

//libraries
#include <boost/python.hpp>
//#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>

//local
#include "eigen_numpy.hpp"
#include "numpy_conversions_shared.hpp"
#include "../math/typedefs.hpp"

// These macros were renamed in NumPy 1.7.1.
#if !defined(NPY_ARRAY_C_CONTIGUOUS) && defined(NPY_C_CONTIGUOUS)
#define NPY_ARRAY_C_CONTIGUOUS NPY_C_CONTIGUOUS
#endif

#if !defined(NPY_ARRAY_ALIGNED) && defined(NPY_ALIGNED)
#define NPY_ARRAY_ALIGNED NPY_ALIGNED
#endif

namespace bp = boost::python;

using namespace Eigen;



template<class TensorType>
struct EigenTensorToPython {
	static PyObject* convert(const TensorType& tensor) {

		const int num_dimensions = TensorType::NumDimensions;
		npy_intp* shape = static_cast<npy_intp*>(malloc(sizeof(npy_intp) * num_dimensions));
		//npy_int shape2[num_dimensions]; //will error, check
		for (int i_dimension = 0; i_dimension < num_dimensions; i_dimension++) {
			shape[i_dimension] = static_cast<npy_intp>(tensor.dimension(i_dimension));
		}

		PyArrayObject* python_array = (PyArrayObject*) PyArray_SimpleNew(
				num_dimensions, shape, NumpyEquivalentType<typename TensorType::Scalar>::type_code);

		copy_tensor(tensor.data(),
				(typename TensorType::Scalar*) PyArray_DATA(python_array),
				num_dimensions,
				shape,
				tensor.size(),
				static_cast<Eigen::StorageOptions>(TensorType::Layout) == Eigen::RowMajor);
		free(shape);
		return (PyObject*) python_array;

	}
};

template<>
struct EigenTensorToPython<math::Tensor3v3f> {
	static PyObject* convert(const math::Tensor3v3f& tensor) {

		const int num_dimensions = math::Tensor3v3f::NumDimensions;
		npy_intp* shape_orig = static_cast<npy_intp*>(malloc(sizeof(npy_intp) * num_dimensions));
		npy_intp* shape_dest = static_cast<npy_intp*>(malloc(sizeof(npy_intp) * (num_dimensions+1)));
		//npy_int shape2[num_dimensions]; //will error, check
		for (int i_dimension = 0; i_dimension < num_dimensions; i_dimension++) {
			shape_orig[i_dimension] = static_cast<npy_intp>(tensor.dimension(i_dimension));
			shape_dest[i_dimension] = static_cast<npy_intp>(tensor.dimension(i_dimension));
		}
		shape_dest[num_dimensions] = 3;

		PyArrayObject* python_array = (PyArrayObject*) PyArray_SimpleNew(
				num_dimensions+1, shape_dest, NumpyEquivalentType<typename math::Tensor3v3f::Scalar>::type_code);

		copy_tensor(tensor.data(),
				(typename math::Tensor3v3f::Scalar*) PyArray_DATA(python_array),
				num_dimensions,
				shape_orig,
				tensor.size(),
				static_cast<Eigen::StorageOptions>(math::Tensor3v3f::Layout) == Eigen::RowMajor);
		free(shape_orig);
		free(shape_dest);
		return (PyObject*) python_array;

	}
};

template<typename TensorType>
struct EigenTensorFromPython {
	typedef typename TensorType::Scalar T;
	EigenTensorFromPython() {
		bp::converter::registry::push_back(&convertible,
				&construct,
				bp::type_id<TensorType>());
	}

	static void* convertible(PyObject* obj_ptr) {
		PyArrayObject* array = reinterpret_cast<PyArrayObject*>(obj_ptr);
		if (!PyArray_Check(array)) {
			//LOG(ERROR) << "PyArray_Check failed";
			return 0;
		}

		int dimension_count = PyArray_NDIM(array);
		if (dimension_count <= 2) {
			//This should be an Eigen::Matrix, not Eigen::Tensor
			//LOG(ERROR) << "PyArray_Check failed";
			return 0;
		} else if (dimension_count == 3) {
			npy_intp* dimensions = PyArray_DIMS(array);
			if (dimensions[2] == 2) {
				//This should be an Eigen::Matrix<Vector2>, not Eigen::Tensor
				return 0;
			}
		} else if (dimension_count != TensorType::NumDimensions) {
			//LOG(ERROR) << "PyArray_Check failed";
			return 0;
		}
		if (PyArray_ObjectType(obj_ptr, 0) != NumpyEquivalentType<T>::type_code) {
			//LOG(ERROR) << "types not compatible";
			return 0;
		}
		int flags = PyArray_FLAGS(array);
		if (!(flags & NPY_ARRAY_C_CONTIGUOUS)) {
			//LOG(ERROR) << "Contiguous C array required";
			return 0;
		}
		if (!(flags & NPY_ARRAY_ALIGNED)) {
			//LOG(ERROR) << "Aligned array required";
			return 0;
		}
		return obj_ptr;
	}

	static void construct(PyObject* obj_ptr,
			bp::converter::rvalue_from_python_stage1_data* data) {

		using bp::extract;

		PyArrayObject* array = reinterpret_cast<PyArrayObject*>(obj_ptr);
		npy_intp* numpy_array_dimensions = PyArray_DIMS(array);

		T* raw_data = reinterpret_cast<T*>(PyArray_DATA(array));

		typedef TensorMap<Tensor<T, TensorType::NumDimensions, RowMajor>, Aligned> TensorMapType;
		typedef TensorLayoutSwapOp<Tensor<T, TensorType::NumDimensions, RowMajor>> TensorSwapLayoutType;

		void* storage = ((bp::converter::rvalue_from_python_storage<TensorType>*)
				(data))->storage.bytes;

		std::array<Index, TensorType::NumDimensions> tensor_dimensions;
		std::array<int, TensorType::NumDimensions> inverse_dimensions;
		for (size_t i_dim = 0, inv_dim = TensorType::NumDimensions - 1; i_dim < tensor_dimensions.size();
				i_dim++, inv_dim--) {
			tensor_dimensions[i_dim] = static_cast<Index>(numpy_array_dimensions[i_dim]);
			inverse_dimensions[i_dim] = inv_dim;
		}

		new (storage) TensorType;
		TensorType* etensor = (TensorType*) storage;

		// TODO: This is a (potentially) expensive copy operation. There should be a better way
		auto mapped_t = TensorMapType(raw_data, tensor_dimensions);
		*etensor = TensorSwapLayoutType(mapped_t).shuffle(inverse_dimensions);
		data->convertible = storage;
	}
};


#if PY_VERSION_HEX >= 0x03000000
void*
#else
void
#endif
setup_Eigen_tensor_converters() {

	static bool is_setup = false;
	if (is_setup)
		return NUMPY_IMPORT_ARRAY_RETVAL;
	is_setup = true;
	import_array();

	EigenTensorFromPython<Eigen::Tensor<float, 3>>();
	bp::to_python_converter<Eigen::Tensor<float, 3>, EigenTensorToPython<Eigen::Tensor<float, 3>>>();

	EigenTensorFromPython<Eigen::Tensor<float, 4>>();
	bp::to_python_converter<Eigen::Tensor<float, 4>, EigenTensorToPython<Eigen::Tensor<float, 4>>>();

	bp::to_python_converter<Eigen::Tensor<float, 3, RowMajor>, EigenTensorToPython<Eigen::Tensor<float, 3, RowMajor>>>();
	bp::to_python_converter<Eigen::Tensor<float, 4, RowMajor>, EigenTensorToPython<Eigen::Tensor<float, 4, RowMajor>>>();
	bp::to_python_converter<math::Tensor3v3f, EigenTensorToPython<math::Tensor3v3f> >();

#if PY_VERSION_HEX >= 0x03000000
	return 0;
#endif
}

