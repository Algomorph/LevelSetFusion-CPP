/*
 * sum.hpp
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

//libraries
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>


namespace math{

template<typename Scalar, typename NestedContainer>
void cwise_nested_sum(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& summed,
		const Eigen::Matrix<NestedContainer, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& field);

template<typename Scalar, typename NestedContainer>
void cwise_nested_sum(Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& summed,
		const Eigen::Tensor<NestedContainer, 3, Eigen::ColMajor>& field);

template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
cwise_square(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& field);

template<typename Scalar>
Eigen::Tensor<Scalar, 3, Eigen::ColMajor>
cwise_square(const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& field);

template<typename Scalar>
inline
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
cwise_abs(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& field){
	return field.array().abs();
}

template<typename Scalar>
inline
Eigen::Tensor<Scalar, 3, Eigen::ColMajor>
cwise_abs(const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& field){
	return field.abs();
}

}  // namespace math
