/*
 * sum.tpp
 *
 *  Created on: Apr 15, 2019
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

//local
#include "cwise_unary.hpp"

namespace math{


template<typename Scalar, typename NestedContainer>
void nested_sum(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& summed,
		const Eigen::Matrix<NestedContainer, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& field){
	summed = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>(field.rows(), field.cols());
#pragma omp parallel for
	for(Eigen::Index i_element = 0; i_element < field.size(); i_element++){
		summed(i_element) = field(i_element).sum();
	}
}

template<typename Scalar, typename NestedContainer>
void nested_sum(Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& summed,
		const Eigen::Tensor<NestedContainer, 3, Eigen::ColMajor>& field){
	summed = Eigen::Tensor<Scalar, 3, Eigen::ColMajor>(field.dimensions());
#pragma omp parallel for
	for(Eigen::Index i_element = 0; i_element < field.size(); i_element++){
		summed(i_element) = field(i_element).sum();
	}
}

}  // namespace math

