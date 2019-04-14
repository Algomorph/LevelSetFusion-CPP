//  ================================================================
//  Created by Gregory Kramida on 10/26/18.
//  Copyright (c) 2018-2019 Gregory Kramida
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

#include "typedefs.hpp"
#include "stacking.hpp"

namespace math {

/**
 * @brief Calculate the 2nd-order discrete finite central differences with h=1 over the vector field and store
 * the results in a vector field.
 * @details Laplacian is the sum of said differences at each location. For edge locations, first-order forward and backward
 * differences are used instead as appropriate.
 * @param[out] laplacian output
 * @param[in] field input
 */
template<typename Scalar>
void laplacian(
		Eigen::Matrix<math::Vector2<Scalar>,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>& laplacian,
		const Eigen::Matrix<math::Vector2<Scalar>,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>& field);
/**
 * @brief Calculate the discrete laplacian (2nd order derivative approximation) over the vector field and store
 * the results in a vector field.
 * @overload
 */
template<typename Scalar>
void laplacian(
		Eigen::Tensor<math::Vector3<Scalar>,3,Eigen::ColMajor>& laplacian,
		const Eigen::Tensor<math::Vector3<Scalar>,3,Eigen::ColMajor>& field);

/**
 * @brief Calculate the negated discrete laplacian over the vector field and store the results in a vector field.
 * @see laplacian for details
 * @param[out] laplacian output
 * @param[in] field input
 */
template<typename Scalar>
void negative_laplacian(
		Eigen::Matrix<math::Vector2<Scalar>,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>& laplacian,
		const Eigen::Matrix<math::Vector2<Scalar>,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>& field);

/**
 * Compute numerical gradient of given field.
 *
 * @details Uses forward and backward differences at the boundary entries,
 * central difference formula for all other entries.
 * Reference: https://en.wikipedia.org/wiki/Numerical_differentiation ;
 * Central difference formula: (f(x + h) - f(x-h))/2h, with h = 1 tensor/matrix grid cell
 *
 * Nested types are determined as follows based on the input field's element type.
 * Input element type:     			Output element type:
 * real (float or double)           Vector2 for 1 output field or real for 2 separate output fields
 * Vector2							Matrix2 (local Jacobian matrix)
 * Vector3							Matrix3 (local Jacobian matrix)
 *
 * @param[out] gradient output field containing gradients.
 * @param[in] field input field.
 */
template<typename Scalar>
void gradient(
		Eigen::Matrix<math::Matrix2<Scalar>,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>& gradient,
		const Eigen::Matrix<math::Vector2<Scalar>,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>& field);

template<typename Scalar>
void gradient(
		Eigen::Tensor<math::Matrix3<Scalar>,3,Eigen::ColMajor>& gradient,
		const Eigen::Tensor<math::Vector3<Scalar>,3,Eigen::ColMajor> field);
/**
 * @brief Calculate the gradient (1st derivative approximation via first-order central differences) of given field.
 * @overload
 */
template<typename Scalar>
void gradient(
		Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>& gradient_x,
		Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>& gradient_y,
		const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>& field);
/**
 * @brief Calculate the gradient (1st derivative approximation via first-order central differences) of given field.
 * @overload
 */
template<typename Scalar>
void gradient(
		Eigen::Matrix<math::Vector2<Scalar>,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>& gradient,
		const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>& field);
/**
 * @brief Calculate the gradient (1st derivative approximation via first-order central differences) of given field.
 * @overload
 */
template<typename Scalar>
void gradient(
		Eigen::Tensor<math::Vector3<Scalar>,3,Eigen::ColMajor>& gradient,
		const Eigen::Tensor<Scalar,3,Eigen::ColMajor>& field);

//an alternative implementation TODO: test which is faster
template<typename Scalar>
void gradient2(
		Eigen::Tensor<math::Vector3<Scalar>,3,Eigen::ColMajor>& gradient,
		const Eigen::Tensor<Scalar,3,Eigen::ColMajor>& field);

}

