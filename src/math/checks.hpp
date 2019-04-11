/*
 * checks.hpp
 *
 *  Created on: Mar 1, 2019
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
#include <unsupported/Eigen/CXX11/Tensor>

namespace math {

inline static
bool is_power_of_two(int number){
	return !(number == 0) && !(number & (number - 1));
}

template<typename ScalarA, typename ScalarB>
inline static
bool are_dimensions_equal(
		const Eigen::Matrix<ScalarA,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>& container_a,
		const Eigen::Matrix<ScalarB,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>& container_b){
	return container_a.rows() == container_a.rows() || container_a.cols() == container_a.cols();
}

template<typename ScalarA, typename ScalarB>
inline static
bool are_dimensions_equal(
		const Eigen::Tensor<ScalarA,3,Eigen::ColMajor>& container_a,
		const Eigen::Tensor<ScalarB,3,Eigen::ColMajor>& container_b){
	for (int i_dim = 0; i_dim < 3; i_dim++) {
		if (container_a.dimension(i_dim) != container_b.dimension(i_dim)) {
			return false;
		}
	}
	return true;
}

} //namespace math


