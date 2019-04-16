/*
 * filtered_statistics.hpp
 *
 *  Created on: Nov 16, 2018
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
//libraries
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

//local
#include "../math/typedefs.hpp"

#pragma once

namespace eig = Eigen;

/**
 * Routines for computing the statistics of fields operating on filtered field regions, e.g. gather statistics only
 * over the span of live & canonical TSDF union or intersection
 */

namespace math {

template<typename Scalar>
double ratio_of_vector_lengths_above_threshold_band_union(
		const Eigen::Matrix<math::Vector2<Scalar>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& vector_field,
		Scalar threshold,
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& warped_live_field,
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& canonical_field);

template<typename Scalar>
double ratio_of_vector_lengths_above_threshold_band_union(
		const Eigen::Tensor<math::Vector3<Scalar>, 3, Eigen::ColMajor>& vector_field,
		Scalar threshold,
		const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& warped_live_field,
		const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& canonical_field);

template<typename Scalar>
void mean_and_std_vector_length_band_union(Scalar& mean, Scalar& standard_deviation,
		const Eigen::Matrix<math::Vector2<Scalar>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& vector_field,
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& warped_live_field,
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& canonical_field);

template<typename Scalar>
void mean_and_std_vector_length_band_union(Scalar& mean, Scalar& standard_deviation,
		const Eigen::Tensor<math::Vector3<Scalar>, 3, Eigen::ColMajor>& vector_field,
		const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& warped_live_field,
		const  Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& canonical_field);

} //namespace math
