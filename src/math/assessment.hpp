/*
 * assesment.hpp
 *
 *  Created on: Feb 1, 2019
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

//stdlib
#include <cmath>
#include <string>
#include <iostream>
#include <cfloat>

//libraries
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace math {

//TODO: split into .hpp/.tpp/.cpp pattern with explicit instantiations

inline bool almost_equal(float a, float b, float epsilon=3e-6) {
	const float absA = std::abs(a);
	const float absB = std::abs(b);
	const float diff = std::abs(a - b);

	//TODO: use FLT_TRUE_MIN instead of FLT_MIN after switching to C++17 or C++20 standard

	if (a == b) { // shortcut, handles infinities
		return true;
	} else if (a == 0 || b == 0 || diff < FLT_MIN) {
		// a or b is zero or both are extremely close to it
		// relative error is less meaningful here
		return diff < (epsilon * FLT_MIN);
	} else { // use relative error
		return diff / std::min((absA + absB), FLT_MAX) < epsilon;
	}
}
//TODO: template almost_equal and propagate its use throughout this file

template<typename TMatrix>
bool matrix_almost_equal(TMatrix matrix_a, TMatrix matrix_b, double tolerance = 1e-10) {
	if (matrix_a.rows() != matrix_b.rows() || matrix_a.cols() != matrix_b.rows()) {
		return false;
	}
	for (Eigen::Index index = 0; index < matrix_a.size(); index++) {
		if (!matrix_a(index).is_close(matrix_b(index), tolerance)) {
			return false;
		}
	}
	return true;
}

template<typename TMatrix>
bool matrix_almost_equal_verbose(TMatrix matrix_a, TMatrix matrix_b, double tolerance = 1e-10) {
	if (matrix_a.rows() != matrix_b.rows() || matrix_a.cols() != matrix_b.rows()) {
		std::cout << "Matrix dimensions don't match. Matrix a: " << matrix_a.cols() << " columns by " << matrix_a.rows()
				<< " rows, Matrix b: " << matrix_b.cols() << " columns by " << matrix_b.rows() << " rows."
				<< std::endl;
		return false;
	}
	for (Eigen::Index index = 0; index < matrix_a.size(); index++) {
		if (!matrix_a(index).is_close(matrix_b(index), tolerance)) {
			long x = index / matrix_a.cols();
			long y = index % matrix_a.cols();
			std::cout << "Matrix entries are not within tolerance threshold of each other. First mismatch at row " << y
					<< ", column " << x << ". " << "Values: " << matrix_a(index) << " vs. " << matrix_b(index)
					<< ", difference: " << matrix_a(index) - matrix_b(index) << std::endl;
			return false;
		}
	}
	return true;
}

template<>
bool matrix_almost_equal_verbose<Eigen::MatrixXf>(Eigen::MatrixXf matrix_a, Eigen::MatrixXf matrix_b,
		double tolerance);

template<typename TEigenContainer, typename LambdaCompareDimensions, typename LambdaCompareElements,
		typename LambdaPrintLocalError>
bool almost_equal(TEigenContainer container_a, TEigenContainer container_b,
		LambdaCompareDimensions&& compare_dimensions,
		LambdaCompareElements&& compare_elements,
		LambdaPrintLocalError&& print_local_error,
		double tolerance
		) {
	std::forward<LambdaCompareDimensions>(compare_dimensions)(container_a, container_b);
	for (Eigen::Index index = 0; index < container_a.size(); index++) {
		if (!std::forward<LambdaCompareElements>(compare_elements)(container_a(index), container_b(index), tolerance)) {
			std::forward<LambdaPrintLocalError>(print_local_error)(container_a, container_b, index);
			return false;
		}
	}
	return true;
}

typedef Eigen::Tensor<float, 3> TenF3;

template<typename TTensor, typename LambdaCompareElements, typename LambdaPrintLocalError>
bool tensor_almost_equal_generic_elements_generic_reporting(TTensor container_a, TTensor container_b,
		LambdaCompareElements&& compare_elements, LambdaPrintLocalError&& print_local_error,
		double tolerance) {
	return almost_equal(container_a, container_b,
			[](TTensor container_a, TTensor container_b)-> bool {
				for (int i_dim = 0; i_dim < TTensor::NumDimensions; i_dim++) {
					if(container_a.dimension(i_dim) != container_b.dimension(i_dim)) {
						std::cout << "Tensor dimension " << i_dim << " (0-based) doesn't match. Tensor a dimension: "
						<< container_a.dimension(i_dim) << ". Corresponding tensor b dimension: " <<
						container_b.dimension(i_dim) << std::endl;
						return false;
					}
				}
				return true;
			},
			compare_elements,
			print_local_error,
			tolerance
			);
}

template<typename TTensor>
inline std::vector<int> unravel_index(TTensor container, Eigen::Index index) {
	std::vector<int> unraveled_index;
	long remainder = static_cast<long>(index);
	for (int i_dimension = 0; i_dimension < TTensor::NumDimensions; i_dimension++) {
		int dimension = container.dimension(i_dimension);
		long new_remainder = remainder / dimension;
		unraveled_index.push_back(remainder % dimension);
		remainder = new_remainder;
	}
	return unraveled_index;
}

template<typename TTensor, typename LambdaCompareElements>
bool tensor_almost_equal_verbose_generic_elements(TTensor container_a, TTensor container_b,
		LambdaCompareElements&& compare_elements, double tolerance) {
	return tensor_almost_equal_generic_elements_generic_reporting(
			container_a, container_b, compare_elements,
			[](TTensor container_a, TTensor container_b, Eigen::Index index) -> void {
				std::vector<int> unraveled_index = unravel_index(container_a, index);
				std::cout << "Tensor elements do not match. First mismatch at linear index " << unraveled_index[0];
				for (size_t ix_index = 1; ix_index < unraveled_index.size(); ix_index++) {
					std::cout << ", " << unraveled_index[ix_index];
				}
				std::cout << ". Values: " << container_a(index) << " vs. " << container_b(index);
			},
			tolerance);
}

template<typename TTensor, typename LambdaCompareElements>
bool tensor_almost_equal_silent_generic_elements(TTensor container_a, TTensor container_b,
		LambdaCompareElements&& compare_elements, double tolerance) {
	return tensor_almost_equal_generic_elements_generic_reporting(container_a, container_b,
			[](TTensor container_a, TTensor container_b, Eigen::Index index) -> void {
			},
			tolerance);
}

template<typename TTensor>
bool tensor_almost_equal_verbose(TTensor container_a, TTensor container_b, double tolerance) {
	return tensor_almost_equal_verbose_generic_elements(container_a, container_b,
			[](typename TTensor::Scalar element_a, typename TTensor::Scalar element_b, double tolerance)-> bool {
				return std::abs(static_cast<double>(element_a) - static_cast<double>(element_b)) < tolerance;
			}, tolerance);
}

template<typename TTensor>
bool tensor_almost_equal(TTensor container_a, TTensor container_b, double tolerance) {
	return tensor_almost_equal_silent_generic_elements(container_a, container_b,
			[](typename TTensor::Scalar element_a, typename TTensor::Scalar element_b, double tolerance)-> bool {
				return std::abs(static_cast<double>(element_a) - static_cast<double>(element_b)) < tolerance;
			}, tolerance);
}

} //namespace math
