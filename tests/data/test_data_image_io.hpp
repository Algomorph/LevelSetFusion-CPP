/*
 * test_data_tsdf.hpp
 *
 *  Created on: Jan 30, 2019
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

#include <Eigen/Eigen>

#include "../../src/math/stacking.hpp"
#include "../src/math/typedefs.hpp"

namespace test_data {
static math::MatrixXus depth_00064_sample = [] {
	math::MatrixXus depth_00064_sample(1,20);
	depth_00064_sample << 2121, 2126, 2124, 2123, 2128, 2133, 2138, 2130, 2135, 2140, 2145,
	2147, 2142, 2147, 2152, 2158, 2150, 2155, 2160, 2165;
	return depth_00064_sample;

}();
} //namespace test_data
