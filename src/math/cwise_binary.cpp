/*
 * multiply.cpp
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

//local
#include "cwise_binary.tpp"
#include "typedefs.hpp"

namespace eig = Eigen;
namespace math {
template eig::MatrixXf scale<float, float>(const eig::MatrixXf& container, float factor);
template eig::MatrixXd scale<double, double>(const eig::MatrixXd& container, double factor);
template math::MatrixXv2f scale<math::Vector2f, float>(const math::MatrixXv2f& container, float factor);
template math::Tensor3f scale<float, float>(const math::Tensor3f& container, float factor);
template math::Tensor3v3f scale<math::Vector3f, float>(const math::Tensor3v3f& container, float factor);

template math::MatrixXv2f cwise_product<math::Vector2f, float>(const math::MatrixXv2f& container_a,
		const eig::MatrixXf& container_b);
template math::Tensor3v3f cwise_product<math::Vector3f, float>(const math::Tensor3v3f& container_a,
		const math::Tensor3f& container_b);

template eig::MatrixXf cwise_add_constant<float>(const eig::MatrixXf& container, float constant);
template math::MatrixXv2f cwise_add_constant<math::Vector2f>(const math::MatrixXv2f& container, math::Vector2f constant);
template math::Tensor3f cwise_add_constant<float>(const math::Tensor3f& container, float constant);
template math::Tensor3v3f cwise_add_constant<math::Vector3f>(const math::Tensor3v3f& container, math::Vector3f constant);

}  // namespace math

