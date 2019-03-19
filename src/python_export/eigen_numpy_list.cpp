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

#include "eigen_numpy.hpp"

//libraries
#include <boost/python.hpp>
#include <Eigen/Eigen>
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>

//local
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

using namespace Eigen;

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

#if PY_VERSION_HEX >= 0x03000000
	return 0;
#endif
}
