/*
 * numpy_conversions_shared.hpp
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

#pragma once


#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/ndarraytypes.h>

template<typename SCALAR>
	struct NumpyEquivalentType {
};

#include "../math/tensors.hpp"
#include "../math/typedefs.hpp"

template<>
struct NumpyEquivalentType<double> {
	enum {
		type_code = NPY_DOUBLE
	};
};
template<>
struct NumpyEquivalentType<float> {
	enum {
		type_code = NPY_FLOAT
	};
};
template<>
struct NumpyEquivalentType<int> {
	enum {
		type_code = NPY_INT
	};
};
template<>
struct NumpyEquivalentType<unsigned short>{
	enum {
		type_code = NPY_USHORT
	};
};
template<>
struct NumpyEquivalentType<unsigned char>{
	enum {
		type_code = NPY_UBYTE
	};
};
template<>
struct NumpyEquivalentType<math::Vector2<float>> {
	enum {
		type_code = NPY_FLOAT
	};
};
template<>
struct NumpyEquivalentType<math::Matrix2<float>> {
	enum {
		type_code = NPY_FLOAT
	};
};
template<>
struct NumpyEquivalentType<std::complex<double> > {
	enum {
		type_code = NPY_CDOUBLE
	};
};
