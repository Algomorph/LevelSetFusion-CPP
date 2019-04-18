/*
 * warp_statistics.hpp
 *
 *  Created on: Nov 14, 2018
 *      Author: Gregory Kramida
 *   Copyright: 2018 Gregory Kramida
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
#include "typedefs.hpp"

namespace math {

/**
 * Locates the maximum L2 norm (length) of the vector in the given field.
 * @param[out] max_norm length of the longest vector
 * @param[out] coordinate the location of the longest vector
 * @param[in] vector_field the vector field to look at
 */
template<typename Scalar>
void locate_max_norm(typename Scalar::Scalar& max_norm, math::Vector2i& coordinates,
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& vector_field);
/**
 * Locates the maximum L2 norm (length) of the vector in the given field.
 * @overload
 */
template<typename Scalar>
void locate_max_norm(typename Scalar::Scalar& max_norm, math::Vector3i& coordinates,
		const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& vector_field);
/**
 * Locates the minimum L2 norm (length) of the vector in the given field.
 * @param[out] max_norm length of the longest vector
 * @param[out] coordinate the location of the longest vector
 * @param[in] vector_field the vector field to look at
 */
template<typename Scalar>
void locate_min_norm(typename Scalar::Scalar& min_norm, math::Vector2i& coordinates,
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& vector_field);

/**
 * Locates the minimum L2 norm (length) of the vector in the given field.
 * @overload
 */
template<typename Scalar>
void locate_min_norm(typename Scalar::Scalar& min_norm, math::Vector3i& coordinates,
		const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& vector_field);

/**
 * Locates the maximum in the given field.
 * @param[out] maximum the maximum coefficient
 * @param[out] coordinates coordinate of the maximum coefficient
 * @param[in] scalar_field the scalar field to look at
 */
template<typename Scalar>
void locate_maximum(Scalar& maximum, math::Vector3i& coordinates,
		const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& scalar_field);

//uses traversal function/functor combo, otherwise the same as locate_max_norm
void locate_max_norm2(float& max_norm, math::Vector2i& coordinates, const math::MatrixXv2f& vector_field);

/**
 * Locate the maximum L2 norm (length) of the vector in the given field.
 * @param[out] min_norm length of the shortest vector
 * @param[out] coordinate the location of the shortest vector
 * @param[in] vector_field the field to look at
 */
template<typename Scalar>
typename Scalar::Scalar max_norm(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& vector_field);
/**
 * Locate the maximum L2 norm (length) of the vector in the given field.
 * @override
 */
template<typename Scalar>
typename Scalar::Scalar max_norm(const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& vector_field);
/**
 * Locate the minimum L2 norm (length) of the vector in the given field.
 * @param[out] min_norm length of the shortest vector
 * @param[out] coordinate the location of the shortest vector
 * @param[in] vector_field the field to look at
 */
template<typename Scalar>
typename Scalar::Scalar min_norm(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& vector_field);
/**
 * Locate the minimum L2 norm (length) of the vector in the given field.
 * @override
 */
template<typename Scalar>
typename Scalar::Scalar min_norm(const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& vector_field);

template<typename Scalar>
Scalar mean(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& field);

template<typename Scalar>
Scalar mean(const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& field);

template<typename Scalar>
Scalar std(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& field);

template<typename Scalar>
Scalar std(const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& field);

float ratio_of_vector_lengths_above_threshold(const math::MatrixXv2f& vector_field, float threshold);
float mean_vector_length(const math::MatrixXv2f& vector_field);
void mean_and_std_vector_length(float& mean, float& standard_deviation, const math::MatrixXv2f& vector_field);

}
