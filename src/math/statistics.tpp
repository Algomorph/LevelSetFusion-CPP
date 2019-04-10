/*
 * statistics.tpp
 *
 *  Created on: Apr 9, 2019
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

#pragma once

//local
#include "statistics.hpp"
#include "tensor_operations.hpp"
#include "vector_operations.hpp"
#include "../traversal/field_traversal_cpu.hpp"
#include "../traversal/index_raveling.hpp"
#include "tensor_operations.hpp"
#include "../error_handling/throw_assert.hpp"

namespace math{

template<typename Scalar>
void locate_max_norm(float& max_norm, math::Vector2i& coordinates,
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& vector_field) {
	float max_squared_norm = 0;
	coordinates = math::Vector2i(0);
	int column_count = static_cast<int>(vector_field.cols());

	for (eig::Index i_element = 0; i_element < vector_field.size(); i_element++) {
		float squared_length = math::squared_norm(vector_field(i_element));
		if (squared_length > max_squared_norm) {
			max_squared_norm = squared_length;
			div_t division_result = div(static_cast<int>(i_element), column_count);
			coordinates.x = division_result.quot;
			coordinates.y = division_result.rem;
		}
	}
	max_norm = std::sqrt(max_squared_norm);
}

template<typename Scalar>
void locate_max_norm(float& max_norm, math::Vector3i& coordinates,
		const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& vector_field){
	int matrix_size = static_cast<int>(vector_field.size());
	int y_stride = vector_field.dimension(0);
	int z_stride = y_stride * vector_field.dimension(1);

	struct NormAndCoordinate {
		NormAndCoordinate() = default;
		NormAndCoordinate(float squared_norm, math::Vector3i coordinate) :
				squared_norm(squared_norm), coordinates(coordinate) {
		}
		float squared_norm = 0.0f;
		math::Vector3i coordinates = math::Vector3i(0);
	};

	std::atomic<NormAndCoordinate> max_norm_and_coordinate { { 0.0f, math::Vector3i(0) } };

#pragma omp parallel for
	for (int i_element = 0; i_element < matrix_size; i_element++) {
		int x, y, z;
		traversal::unravel_3d_index(x, y, z, i_element, y_stride, z_stride);
		NormAndCoordinate last_max_norm = max_norm_and_coordinate.load(); // @suppress("Invalid arguments")
		NormAndCoordinate new_norm;
		do {
			new_norm = last_max_norm;
			new_norm.squared_norm = math::squared_norm(vector_field(i_element));
			new_norm.coordinates = math::Vector3i(x, y, z);
		} while (new_norm.squared_norm > last_max_norm.squared_norm &&
				!max_norm_and_coordinate.compare_exchange_strong(last_max_norm, new_norm));
	}
	NormAndCoordinate last_max_norm = max_norm_and_coordinate.load(); // @suppress("Invalid arguments")
	max_norm = std::sqrt(last_max_norm.squared_norm);
	coordinates = last_max_norm.coordinates;
}

template<typename Scalar>
void locate_min_norm(float& min_norm, math::Vector2i& coordinates,
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& vector_field) {
	int column_count = static_cast<int>(vector_field.cols());
	struct NormLoc {
		NormLoc() = default;
		float norm = 1000000.0f;
		math::Vector2i coordinates = math::Vector2i(0);
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
			new_best.coordinates.x = i_element / column_count;
			new_best.coordinates.y = i_element % column_count;
		} while (squared_length < min_squared_norm && min_norm_container.compare_exchange_strong(last_best, new_best));
	}
	NormLoc last_best = min_norm_container.load();
	min_norm = std::sqrt(last_best.norm);
	coordinates = last_best.coordinates;
}

/**
 * * Locates the minimum L2 norm (length) of the vector in the given field.
 * @overload
 */
template<typename Scalar>
void locate_min_norm(float& min_norm, math::Vector3i& coordinates,
		const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& vector_field){
	int matrix_size = static_cast<int>(vector_field.size());
	int y_stride = vector_field.dimension(0);
	int z_stride = y_stride * vector_field.dimension(1);

	struct NormAndCoordinate {
		NormAndCoordinate() = default;
		NormAndCoordinate(float squared_norm, math::Vector3i coordinate) :
				squared_norm(squared_norm), coordinates(coordinate) {
		}
		float squared_norm = 0.0f;
		math::Vector3i coordinates = math::Vector3i(0);
	};

	std::atomic<NormAndCoordinate> min_norm_and_coordinate { { 0.0f, math::Vector3i(0) } };

#pragma omp parallel for
	for (int i_element = 0; i_element < matrix_size; i_element++) {
		int x, y, z;
		traversal::unravel_3d_index(x, y, z, i_element, y_stride, z_stride);
		NormAndCoordinate last_min_norm = min_norm_and_coordinate.load(); // @suppress("Invalid arguments")
		NormAndCoordinate new_norm;
		do {
			new_norm = last_min_norm;
			new_norm.squared_norm = math::squared_norm(vector_field(i_element));
			new_norm.coordinates = math::Vector3i(x, y, z);
		} while (new_norm.squared_norm < last_min_norm.squared_norm &&
				!min_norm_and_coordinate.compare_exchange_strong(last_min_norm, new_norm));
	}
	NormAndCoordinate last_max_norm = min_norm_and_coordinate.load(); // @suppress("Invalid arguments")
	min_norm = std::sqrt(last_max_norm.squared_norm);
	coordinates = last_max_norm.coordinates;
}

template<typename Container>
static inline
float max_norm_aux(const Container& vector_field) {
	std::atomic<float> max_norm_container(0.0f);
#pragma omp parallel for
	for (eig::Index i_element = 0; i_element < vector_field.size(); i_element++) {
		float squared_length = math::squared_norm(vector_field(i_element));
		float max_squared_norm = max_norm_container.load();
		while (squared_length > max_squared_norm
				&& max_norm_container.compare_exchange_weak(max_squared_norm, squared_length))
			;
	}
	return std::sqrt(max_norm_container.load());
}

template<typename Scalar>
float max_norm(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& vector_field) {
	return max_norm_aux<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >(vector_field);
}

template<typename Scalar>
float max_norm(const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& vector_field){
	return max_norm_aux<Eigen::Tensor<Scalar, 3, Eigen::ColMajor> > (vector_field);
}

template<typename Container>
static inline
float min_norm_aux(const Container& vector_field){
	std::atomic<float> min_norm_container(std::numeric_limits<float>::max());
	#pragma omp parallel for
	for (eig::Index i_element = 0; i_element < vector_field.size(); i_element++) {
		float squared_length = math::squared_norm(vector_field(i_element));
		float min_squared_norm = min_norm_container.load();
		while (squared_length < min_squared_norm
				&& min_norm_container.compare_exchange_weak(min_squared_norm, squared_length))
			;
	}
	return std::sqrt(min_norm_container.load());
}

template<typename Scalar>
float min_norm(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& vector_field) {
	return min_norm_aux<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >(vector_field);
}

template<typename Scalar>
float min_norm(const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& vector_field){
	return min_norm_aux<Eigen::Tensor<Scalar, 3, Eigen::ColMajor> >(vector_field);
}


} //namespace math


