/*
 * sum.cpp
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

//local
#include "cwise_unary.tpp"
#include "typedefs.hpp"

namespace eig = Eigen;

namespace math{
template void cwise_nested_sum<float,math::Vector2f>(eig::MatrixXf& summed, const math::MatrixXv2f& field);
template void cwise_nested_sum<float,math::Matrix2f>(eig::MatrixXf& summed, const math::MatrixXm2f& field);
template void cwise_nested_sum<float,math::Vector3f>(math::Tensor3f& summed, const math::Tensor3v3f& field);
template void cwise_nested_sum<float,math::Matrix3f>(math::Tensor3f& summed, const math::Tensor3m3f& field);
template eig::MatrixXf cwise_square(const eig::MatrixXf& field);
template math::Tensor3f  cwise_square(const math::Tensor3f& field);

}  // namespace math
