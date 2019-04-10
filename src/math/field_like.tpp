/*
 * field_like.tpp
 *
 *  Created on: Apr 10, 2019
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

//local
#include "field_like.hpp"

namespace math{

template<typename ScalarIn, typename ScalarOut>
Eigen::Matrix<ScalarOut, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
field_like(const Eigen::Matrix<ScalarIn, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& field){
	return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>(field.rows(), field.cols());
}

template<typename ScalarIn, typename ScalarOut>
Eigen::Tensor<ScalarOut, 3, Eigen::ColMajor>
field_like(const Eigen::Tensor<ScalarIn, 3, Eigen::ColMajor>& field){
	return Eigen::Tensor<ScalarOut, 3, Eigen::ColMajor>(field.dimensions());
}

} //namespace math


