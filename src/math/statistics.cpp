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

//local
#include "statistics.hpp"
#include "vector_operations.hpp"
#include "../traversal/field_traversal_cpu.hpp"
#include "../traversal/index_raveling.hpp"
#include "tensor_operations.hpp"

namespace math {

static inline void check_dimensions(const eig::MatrixXf& warped_live_field, const eig::MatrixXf& canonical_field) {
	eigen_assert( canonical_field.rows() == warped_live_field.rows() &&
			canonical_field.cols() == warped_live_field.cols() &&
			"Dimensions of one of the input matrices don't seem to match.");
}

/**
 * Locates the maximum L2 norm (length) of the vector in the given field.
 * @param[out] max_norm length of the longest vector
 * @param[out] coordinate the location of the longest vector
 * @param vector_field the field to look at
 */
void locate_max_norm(float& max_norm, Vector2i& coordinate, const MatrixXv2f& vector_field) {
	float max_squared_norm = 0;
	coordinate = math::Vector2i(0);
	int column_count = static_cast<int>(vector_field.cols());

	for (eig::Index i_element = 0; i_element < vector_field.size(); i_element++) {
		float squared_length = math::squared_norm(vector_field(i_element));
		if (squared_length > max_squared_norm) {
			max_squared_norm = squared_length;
			div_t division_result = div(static_cast<int>(i_element), column_count);
			coordinate.x = division_result.quot;
			coordinate.y = division_result.rem;
		}
	}
	max_norm = std::sqrt(max_squared_norm);
}

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
 * Locates the minimum L2 norm (length) of the vector in the given field.
 * @param[out] min_norm length of the shortest vector
 * @param[out] coordinate the location of the shortest vector
 * @param vector_field the field to look at
 */
void locate_min_norm(float& min_norm, Vector2i& coordinate, const MatrixXv2f& vector_field) {
	int column_count = static_cast<int>(vector_field.cols());
	struct NormLoc {
		NormLoc() = default;
		float norm = 1000000.0f;
		math::Vector2i coordinate = math::Vector2i(0);
	};
	NormLoc initial;
	std::atomic<NormLoc> min_norm_container(initial);

#pragma omp parallel for
	for (eig::Index i_element = 0; i_element < vector_field.size(); i_element++) {
		float squared_length = math::squared_norm(vector_field(i_element));
		NormLoc last_best = min_norm_container.load();
		NormLoc new_best;
		float min_squared_norm = last_best.norm;
		do {
			new_best = last_best;
			new_best.norm = squared_length;
			new_best.coordinate.x = i_element / column_count;
			new_best.coordinate.y = i_element % column_count;
		} while (squared_length < min_squared_norm && min_norm_container.compare_exchange_strong(last_best, new_best));
	}
	NormLoc last_best = min_norm_container.load();
	min_norm = std::sqrt(last_best.norm);
	coordinate = last_best.coordinate;
}

/**
 * Locates the minimum L2 norm (length) of the vector in the given field.
 * @param[out] min_norm length of the shortest vector
 * @param[out] coordinate the location of the shortest vector
 * @param vector_field the field to look at
 */
float min_norm(const MatrixXv2f& vector_field) {
	std::atomic<float> min_norm_container(0.0f);
#pragma omp parallel for
	for (eig::Index i_element = 0; i_element < vector_field.size(); i_element++) {
		float squared_length = math::squared_norm(vector_field(i_element));
		float min_squared_norm = min_norm_container.load();
		while (squared_length < min_squared_norm && min_norm_container.compare_exchange_weak(min_squared_norm, squared_length));
	}
	return std::sqrt(min_norm_container.load());
}
/**
* Locates the maximum L2 norm (Euclidean length) of the vector in the given 3d field.
* Identical to @see locate_max_norm with the exception that this version is using a generic traversal function.
* @param[out] max_norm length of the longest vector
* @param[out] coordinate the location of the longest vector
* @param[in] vector_field the field to look at
*/
void locate_max_norm_3d(float& max_norm, math::Vector3i& coordinates, const math::Tensor3v3f& vector_field)  {
	int matrix_size = static_cast<int>(vector_field.size());
	int y_stride = vector_field.dimension(0);
	int z_stride = y_stride * vector_field.dimension(1);

	struct NormAndCoordinate {
		NormAndCoordinate() = default;
		NormAndCoordinate(float squared_norm, math::Vector3i coordinate):
			squared_norm(squared_norm),coordinates(coordinate){}
		float squared_norm = 0.0f;
		math::Vector3i coordinates = math::Vector3i(0);
	};

	std::atomic<NormAndCoordinate> max_norm_and_coordinate{{0.0f,math::Vector3i(0)}};

#pragma omp parallel for
	for (int i_element = 0; i_element < matrix_size; i_element++) {
		int x, y, z;
		traversal::unravel_3d_index(x, y, z, i_element, y_stride, z_stride);
		NormAndCoordinate last_max_norm = max_norm_and_coordinate.load();
		NormAndCoordinate new_norm;
		do{
			new_norm = last_max_norm;
			new_norm.squared_norm = math::squared_norm(vector_field(i_element));
			new_norm.coordinates = math::Vector3i(x,y,z);
		} while (new_norm.squared_norm > last_max_norm.squared_norm &&
				!max_norm_and_coordinate.compare_exchange_strong(last_max_norm, new_norm));
	}
	NormAndCoordinate last_max_norm = max_norm_and_coordinate.load();
	max_norm = std::sqrt(last_max_norm.squared_norm);
	coordinates = last_max_norm.coordinates;
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
