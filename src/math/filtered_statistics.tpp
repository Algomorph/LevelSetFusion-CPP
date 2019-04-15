/*
 * filtered_statistics.cpp
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

//stdlib
#include <atomic>

//local
#include "filtered_statistics.hpp"
#include "boolean_operations.hpp"
#include "vector_operations.hpp"
#include "typedefs.hpp"
#include "../error_handling/throw_assert.hpp"
#include "checks.hpp"

namespace math {

template<typename VectorContainer, typename ScalarContainer, typename Scalar>
static inline
double ratio_of_vector_lengths_above_threshold_band_union_aux(
		VectorContainer vector_field,
		Scalar threshold,
		ScalarContainer warped_live_field,
		ScalarContainer canonical_field) {

	throw_assert(math::are_dimensions_equal(vector_field, warped_live_field, canonical_field),
			"Dimensions of one of the input matrices don't appear to match.");

	Scalar threshold_squared = threshold * threshold;

	std::atomic<long> count_above(0);
	std::atomic<long> total_count(0);

#pragma omp parallel for
	for (eig::Index i_element = 0; i_element < vector_field.size(); i_element++) {
		if (are_both_SDF_values_truncated(warped_live_field(i_element), canonical_field(i_element)))
			continue;
		float squared_length = math::squared_norm(vector_field(i_element));
		if (squared_length > threshold_squared) {
			count_above++;
		}
		total_count++;
	}
	return static_cast<double>(count_above) / static_cast<double>(total_count);
}

template<typename Scalar>
double ratio_of_vector_lengths_above_threshold_band_union(
		const Eigen::Matrix<math::Vector2<Scalar>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& vector_field,
		Scalar threshold,
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& warped_live_field,
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& canonical_field) {
	return ratio_of_vector_lengths_above_threshold_band_union_aux(vector_field, threshold, warped_live_field,
			canonical_field);
}

template<typename Scalar>
double ratio_of_vector_lengths_above_threshold_band_union(
		const Eigen::Tensor<math::Vector3<Scalar>, 3, Eigen::ColMajor>& vector_field,
		Scalar threshold,
		const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& warped_live_field,
		const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& canonical_field) {
	return ratio_of_vector_lengths_above_threshold_band_union_aux(vector_field, threshold, warped_live_field,
			canonical_field);
}

template<typename VectorContainer, typename ScalarContainer, typename Scalar>
static inline
void mean_and_std_vector_length_band_union_aux(
		Scalar& mean, Scalar& standard_deviation,
		VectorContainer vector_field,
		ScalarContainer warped_live_field,
		ScalarContainer canonical_field) {

	throw_assert(math::are_dimensions_equal(vector_field, warped_live_field, canonical_field),
				"Dimensions of one of the input matrices don't appear to match.");

		std::atomic<long> total_count(0);
		std::atomic<double> total_length(0.0);

	#pragma omp parallel for
		for (eig::Index i_element = 0; i_element < vector_field.size(); i_element++) {
			if (are_both_SDF_values_truncated(warped_live_field(i_element), canonical_field(i_element)))
				continue;
			Scalar length = math::length(vector_field(i_element));
			double expected_total_length = total_length.load();
			double new_total_length;
			do {
				new_total_length = expected_total_length + length;
			} while (!total_length.compare_exchange_weak(expected_total_length, new_total_length));
			total_count += 1;
		}

		mean = static_cast<Scalar>(total_length.load() / static_cast<double>(total_count.load()));
		std::atomic<double> total_squared_deviation(0.0);
	#pragma omp parallel for
		for (eig::Index i_element = 0; i_element < vector_field.size(); i_element++) {
			if (are_both_SDF_values_truncated(warped_live_field(i_element), canonical_field(i_element)))
				continue;
			Scalar length = math::length(vector_field(i_element));
			Scalar local_deviation = length - mean;
			local_deviation = local_deviation * local_deviation;
			double expected_total_squared_deviation = total_squared_deviation.load();
			double new_total_squared_deviation;
			do {
				new_total_squared_deviation = expected_total_squared_deviation + local_deviation;
			} while (!total_squared_deviation.compare_exchange_weak(expected_total_squared_deviation,
					new_total_squared_deviation));
		}
		standard_deviation = static_cast<Scalar>(std::sqrt(
				total_squared_deviation.load() / static_cast<double>(total_count.load())));
}

template<typename Scalar>
void mean_and_std_vector_length_band_union(Scalar& mean, Scalar& standard_deviation,
		const Eigen::Matrix<math::Vector2<Scalar>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& vector_field,
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& warped_live_field,
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& canonical_field) {
	mean_and_std_vector_length_band_union_aux(mean, standard_deviation, vector_field, warped_live_field, canonical_field);
}

template<typename Scalar>
void mean_and_std_vector_length_band_union(Scalar& mean, Scalar& standard_deviation,
		const Eigen::Tensor<math::Vector3<Scalar>, 3, Eigen::ColMajor>& vector_field,
		const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& warped_live_field,
		const  Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& canonical_field) {
	mean_and_std_vector_length_band_union_aux(mean, standard_deviation, vector_field, warped_live_field, canonical_field);
}

} //namespace math
