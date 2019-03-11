/*
 * test.cpp
 *
 *  Created on: Mar 9, 2019
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

//DEBUG
//TODO: remove the whole file & usages of functions/classes within, and also usage in Python code:
// this is testing/debugging code
#pragma once

//libraries
#include <Eigen/Eigen>
#include <boost/python.hpp>
#include "eigen_numpy.hpp"

#include "../nonrigid_optimization/hierarchical_optimizer2d.hpp"

namespace eig = Eigen;
namespace bp = boost::python;

namespace python_export {
namespace test_code {
class TestClass {
public:
	TestClass(eig::VectorXf pmv) :
			private_member_vector(pmv) {
	}

	private:
	eig::VectorXf private_member_vector;
};

void export_test_class() {
	bp::class_<TestClass>("TestClass", bp::init<eig::VectorXf>(bp::args("private_vector_member")));
}

} //namespace test_code
} //namespace python_export
