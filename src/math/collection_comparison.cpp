/*
 * assessment.cpp
 *
 *  Created on: Mar 15, 2019
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
#include <Eigen/Eigen>

//local
#include "collection_comparison.tpp"
#include "typedefs.hpp"

namespace eig = Eigen;
namespace math {

template bool matrix_almost_equal<eig::MatrixXf, float>(eig::MatrixXf matrix_a, eig::MatrixXf matrix_b,
		float tolerance);
template bool matrix_almost_equal<eig::MatrixXf, double>(eig::MatrixXf matrix_a, eig::MatrixXf matrix_b,
		double tolerance);
template bool matrix_almost_equal_verbose<eig::MatrixXf, float>(eig::MatrixXf matrix_a, eig::MatrixXf matrix_b,
		float tolerance);
template bool matrix_almost_equal_verbose<eig::MatrixXf, double>(eig::MatrixXf matrix_a, eig::MatrixXf matrix_b,
		double tolerance);

template bool matrix_almost_equal<math::MatrixXv2f, float>(math::MatrixXv2f matrix_a, math::MatrixXv2f matrix_b,
		float tolerance);
template bool matrix_almost_equal<math::MatrixXv2f, double>(math::MatrixXv2f matrix_a,
		math::MatrixXv2f matrix_b, double tolerance);
template bool matrix_almost_equal_verbose<math::MatrixXv2f, float>(math::MatrixXv2f matrix_a, math::MatrixXv2f matrix_b,
		float tolerance);
template bool matrix_almost_equal_verbose<math::MatrixXv2f, double>(math::MatrixXv2f matrix_a,
		math::MatrixXv2f matrix_b, double tolerance);

template bool matrix_almost_equal<math::MatrixXm2f, float>(math::MatrixXm2f matrix_a, math::MatrixXm2f matrix_b,
		float tolerance);
template bool matrix_almost_equal<math::MatrixXm2f, double>(math::MatrixXm2f matrix_a,
		math::MatrixXm2f matrix_b, double tolerance);
template bool matrix_almost_equal_verbose<math::MatrixXm2f, float>(math::MatrixXm2f matrix_a, math::MatrixXm2f matrix_b,
		float tolerance);
template bool matrix_almost_equal_verbose<math::MatrixXm2f, double>(math::MatrixXm2f matrix_a,
		math::MatrixXm2f matrix_b, double tolerance);

template bool tensor_almost_equal<math::Tensor3f, float>(math::Tensor3f tensor_a, math::Tensor3f tensor_b,
		float tolerance);
template bool tensor_almost_equal<math::Tensor3f, double>(math::Tensor3f tensor_a, math::Tensor3f tensor_b,
		double tolerance);
template bool tensor_almost_equal_verbose<math::Tensor3f, float>(math::Tensor3f tensor_a, math::Tensor3f tensor_b,
		float tolerance);
template bool tensor_almost_equal_verbose<math::Tensor3f, double>(math::Tensor3f tensor_a, math::Tensor3f tensor_b,
		double tolerance);

template bool tensor_almost_equal<math::Tensor3v3f, float>(math::Tensor3v3f tensor_a, math::Tensor3v3f tensor_b,
		float tolerance);
template bool tensor_almost_equal<math::Tensor3v3f, double>(math::Tensor3v3f tensor_a, math::Tensor3v3f tensor_b,
		double tolerance);
template bool tensor_almost_equal_verbose<math::Tensor3v3f, float>(math::Tensor3v3f tensor_a, math::Tensor3v3f tensor_b,
		float tolerance);
template bool tensor_almost_equal_verbose<math::Tensor3v3f, double>(math::Tensor3v3f tensor_a, math::Tensor3v3f tensor_b,
		double tolerance);

template bool almost_equal<float>(float a, float b);
template bool almost_equal<float, float>(float a, float b, float tolerance);
template bool almost_equal<float, double>(float a, float b, double tolerance);
template bool almost_equal<double, double>(double a, double b, double tolerance);
template bool almost_equal<math::Vector2f, float>(math::Vector2f a, math::Vector2f b, float tolerance);
template bool almost_equal<math::Vector2f, double>(math::Vector2f a, math::Vector2f b, double tolerance);
template bool almost_equal<math::Vector3f, float>(math::Vector3f a, math::Vector3f b, float tolerance);
template bool almost_equal<math::Vector3f, double>(math::Vector3f a, math::Vector3f b, double tolerance);

} //namespace math

