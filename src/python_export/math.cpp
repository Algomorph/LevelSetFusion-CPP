/*
 * math.cpp
 *
 *  Created on: Jan 11, 2019
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

//libraries
#include <Eigen/Eigen>
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

//local
#include "../math/typedefs.hpp"

namespace bp = boost::python;

namespace python_export {

void export_math_types() {
	// =============================================================================================
	// region === MATH TYPES =======================================================================
	// =============================================================================================
	bp::class_<math::Vector2i>("Vector2i", bp::init<int, int>())
			.def(bp::init<int>())
			.def_readwrite("x", &math::Vector2i::x)
			.def_readwrite("y", &math::Vector2i::y)
			.def_readwrite("u", &math::Vector2i::u)
			.def_readwrite("v", &math::Vector2i::v)
			;

	bp::class_<math::Vector2f>("Vector2f", bp::init<float, float>())
			.def(bp::init<float>())
			.def_readwrite("x", &math::Vector2f::x)
			.def_readwrite("y", &math::Vector2f::y)
			.def_readwrite("u", &math::Vector2f::u)
			.def_readwrite("v", &math::Vector2f::v)
			;
}

} // namespace python_export

