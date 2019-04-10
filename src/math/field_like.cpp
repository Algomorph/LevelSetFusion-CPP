/*
 * field_like.cpp
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


//libraries
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

//local
#include "field_like.tpp"
#include "typedefs.hpp"

namespace eig = Eigen;

namespace math{

template eig::MatrixXf field_like<float, float>(const eig::MatrixXf& field);
template math::MatrixXv2f field_like<float, math::Vector2f>(const eig::MatrixXf& field);
template math::MatrixXv2f field_like<math::Vector2f, math::Vector2f>(const math::MatrixXv2f& field);

template math::Tensor3f field_like<float, float>(const math::Tensor3f& field);
template math::Tensor3v3f field_like<float, math::Vector3f>(const math::Tensor3f& field);
template math::Tensor3v3f field_like<math::Vector3f, math::Vector3f>(const math::Tensor3v3f& field);

} //namespace math


