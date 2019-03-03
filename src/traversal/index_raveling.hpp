/*
 * index_raveling.hpp
 *
 *  Created on: Mar 3, 2019
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
#include <Eigen/Dense>

inline static void unravel_3d_index(int& x, int& y, int& z, Eigen::Index i_element,
		const int& y_stride, const int& z_stride){
	int z_field = i_element / z_stride;
	int remainder = i_element % z_stride;
	int y_field = remainder / y_stride;
	int x_field = remainder % y_stride;
}



