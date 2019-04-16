//  ================================================================
//  Created by Gregory Kramida on 10/23/18.
//  Copyright (c) 2018 Gregory Kramida
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//  ================================================================
#pragma once

//stdlib
#include <cstdlib>
#include <vector>

//libraries
#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>

//local
#include "vector2.hpp"
#include "matrix2.hpp"
#include "typedefs.hpp"


namespace math {


/***
 * Make a nested vector field out of a set of scalar fields by stacking values in each location into a vector.
 * @param set the set of scalar fields to use
 */
template<typename Scalar>
Eigen::Matrix<math::Vector2<Scalar>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
stack(const std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>& set);

template<typename Scalar>
Eigen::Tensor<Scalar, 3, Eigen::ColMajor>
stack(const std::vector<Eigen::Tensor<Scalar, Eigen::ColMajor>>& set);

//legacy
MatrixXv2f stack_as_xv2f(const Eigen::MatrixXf& matrix_a, const Eigen::MatrixXf& matrix_b);
void unstack_xv2f(Eigen::MatrixXf& matrix_a, Eigen::MatrixXf& matrix_b, const MatrixXv2f vector_field);

} //namespace math


