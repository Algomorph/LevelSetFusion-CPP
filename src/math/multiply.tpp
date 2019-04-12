/*
 * multiply.tpp
 *
 *  Created on: Apr 11, 2019
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

//libraries
#include <Eigen/Eigen>

//local
#include "multiply.hpp"
#include "checks.hpp"
#include "../error_handling/throw_assert.hpp"

namespace math {

template<typename ScalarMajor, typename ScalarMinor>
Eigen::Matrix<ScalarMajor, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
scale(const Eigen::Matrix<ScalarMajor, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& container, ScalarMinor factor){
	return factor * container;
}

template<typename ScalarMajor, typename ScalarMinor>
Eigen::Tensor<ScalarMajor, 3, Eigen::ColMajor>
scale(const Eigen::Tensor<ScalarMajor, 3, Eigen::ColMajor>& container, ScalarMinor factor){
	Eigen::Tensor<ScalarMajor, 3, Eigen::ColMajor> result = Eigen::Tensor<ScalarMajor, 3, Eigen::ColMajor>(container.dimensions());
#pragma omp parallel for
	for (Eigen::Index i_element = 0; i_element < container.size(); i_element++){
		result(i_element) = container(i_element) * factor;
	}
	return result;
}

template<typename ScalarMajor, typename ScalarMinor>
Eigen::Matrix<ScalarMajor, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
cwise_product(
		const Eigen::Matrix<ScalarMajor, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& container_a,
		const Eigen::Matrix<ScalarMinor, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& container_b){
	return container_a.cwiseProduct(container_b);
}



template<typename ScalarMajor, typename ScalarMinor>
Eigen::Tensor<ScalarMajor, 3, Eigen::ColMajor>
cwise_product(
		const Eigen::Tensor<ScalarMajor, 3, Eigen::ColMajor>& container_a,
		const Eigen::Tensor<ScalarMinor, 3, Eigen::ColMajor>& container_b){

	throw_assert(math::are_dimensions_equal(container_a,container_b), "Tensor dimensions differ.");
	Eigen::Tensor<ScalarMajor, 3, Eigen::ColMajor> result = Eigen::Tensor<ScalarMajor, 3, Eigen::ColMajor>(container_a.dimensions());

#pragma omp parallel for
	for (Eigen::Index i_element = 0; i_element < container_a.size(); i_element++){
		result(i_element) = container_a(i_element) * container_b(i_element);
	}
	return result;
}


}  // namespace math


