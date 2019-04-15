/*
 * cwise_binary.hpp
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
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace math {

template<typename ScalarMajor, typename ScalarMinor>
Eigen::Matrix<ScalarMajor, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
scale(const Eigen::Matrix<ScalarMajor, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& container, ScalarMinor factor);

template<typename ScalarMajor, typename ScalarMinor>
Eigen::Tensor<ScalarMajor, 3, Eigen::ColMajor>
scale(const Eigen::Tensor<ScalarMajor, 3, Eigen::ColMajor>& container, ScalarMinor factor);

template<typename ScalarMajor, typename ScalarMinor>
Eigen::Matrix<ScalarMajor, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
cwise_product(
		const Eigen::Matrix<ScalarMajor, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& container_a,
		const Eigen::Matrix<ScalarMinor, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& container_b);

template<typename ScalarMajor, typename ScalarMinor>
Eigen::Tensor<ScalarMajor, 3, Eigen::ColMajor>
cwise_product(
		const Eigen::Tensor<ScalarMajor, 3, Eigen::ColMajor>& container_a,
		const Eigen::Tensor<ScalarMinor, 3, Eigen::ColMajor>& container_b);

template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
cwise_add_constant(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& container, Scalar constant);

template<typename Scalar>
Eigen::Tensor<Scalar, 3, Eigen::ColMajor>
cwise_add_constant(const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& container, Scalar constant);

}  // namespace math


