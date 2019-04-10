/*
 * statistics.cpp
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

//stdlib
#include <cmath>
#include <atomic>
#include <limits>

//local
#include "statistics.tpp"


namespace math {

template void locate_max_norm<math::Vector2f>(float& max_norm, math::Vector2i& coordinates, const math::MatrixXv2f& vector_field);
template void locate_max_norm<math::Vector3f>(float& max_norm, math::Vector3i& coordinates, const math::Tensor3v3f& vector_field);
template void locate_min_norm<math::Vector2f>(float& min_norm, math::Vector2i& coordinates, const math::MatrixXv2f& vector_field);
template void locate_min_norm<math::Vector3f>(float& min_norm, math::Vector3i& coordinates, const math::Tensor3v3f& vector_field);

template float max_norm<math::Vector2f>(const math::MatrixXv2f& vector_field);
template float max_norm<math::Vector3f>(const math::Tensor3v3f& vector_field);
template float min_norm<math::Vector2f>(const math::MatrixXv2f& vector_field);
template float min_norm<math::Vector3f>(const math::Tensor3v3f& vector_field);

/**
 * Locates the maximum L2 norm (length) of the vector in the given field.
 * Identical to @see locate_max_norm with the exception that this version is using a generic traversal function.
 * @param[out] max_norm length of the longest vector
 * @param[out] coordinate the location of the longest vector
 * @param[in] vector_field the field to look at
 */
void locate_max_norm2(float& max_norm, Vector2i& coordinate, const MatrixXv2f& vector_field) {
	float max_squared_norm = 0;
	coordinate = math::Vector2i(0);
	int column_count = static_cast<int>(vector_field.cols());

	auto max_norm_functor = [&] (math::Vector2f element, eig::Index i_element) {
		float squared_length = math::squared_norm(element);
		if(squared_length > max_squared_norm) {
			max_squared_norm = squared_length;
			div_t division_result = div(static_cast<int>(i_element), column_count);
			coordinate.x = division_result.quot;
			coordinate.y = division_result.rem;
		}
	};
	traversal::traverse_2d_field_i_element_singlethreaded(vector_field, max_norm_functor);
	max_norm = std::sqrt(max_squared_norm);
}


/**
 * Given a 2d vector field, computes the lengths and counts up how many of the vectors are above the provided threshold
 * @param vector_field - vector field to look at
 * @param threshold - an arbitrary threshold
 * @return ratio of the counted vectors to the total number of vectors in the field
 */
float ratio_of_vector_lengths_above_threshold(const MatrixXv2f& vector_field, float threshold) {
	float threshold_squared = threshold * threshold;
	long count_above = 0;
	long total_count = vector_field.size();

	for (eig::Index i_element = 0; i_element < vector_field.size(); i_element++) {
		float squared_length = math::squared_norm(vector_field(i_element));
		if (squared_length > threshold_squared) {
			count_above++;
		}
	}
	return static_cast<double>(count_above) / static_cast<double>(total_count);
}

/**
 * Computes the mean vector length for all vectors in the given field
 * @param vector_field the field to look at
 * @return the arithmetic mean vector length
 */
float mean_vector_length(const MatrixXv2f& vector_field) {
	long total_count = vector_field.size();
	double total_length = 0.0;

	for (eig::Index i_element = 0; i_element < vector_field.size(); i_element++) {
		float length = math::length(vector_field(i_element));
		total_length += static_cast<double>(length);
	}
	return static_cast<float>(total_length / static_cast<double>(total_count));
}

/**
 * Computes the mean vector length and the standard deviation of lengths for vectors in the given field
 * @param[out] mean the arithmetic mean vector length
 * @param[out] standard_deviation standard deviation of vector lengths
 * @param[in] vector_field the field to look at
 */
void mean_and_std_vector_length(float& mean, float& standard_deviation, const MatrixXv2f& vector_field) {
	long total_count = vector_field.size();
	double total_length = 0.0;

	for (eig::Index i_element = 0; i_element < vector_field.size(); i_element++) {
		float length = math::length(vector_field(i_element));
		total_length += static_cast<double>(length);
	}

	mean = static_cast<float>(total_length / static_cast<double>(total_count));
	double total_squared_deviation = 0.0;
	for (eig::Index i_element = 0; i_element < vector_field.size(); i_element++) {
		float length = math::length(vector_field(i_element));
		float local_deviation = length - mean;
		total_squared_deviation += local_deviation * local_deviation;
	}
	standard_deviation = static_cast<float>(std::sqrt(total_squared_deviation / static_cast<double>(total_count)));
}

} //namespace math
