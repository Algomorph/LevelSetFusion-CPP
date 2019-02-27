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

//libraries
#include <Eigen/Dense>

//DEBUG
#include <unsupported/Eigen/CXX11/Tensor>

namespace math {
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
			ldiv_t division_result = div(index, matrix_a.cols());
			long x = division_result.quot;
			long y = division_result.rem;
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
		double tolerance) {
	if (matrix_a.rows() != matrix_b.rows() || matrix_a.cols() != matrix_b.rows()) {
		std::cout << "Matrix dimensions don't match. Matrix a: " << matrix_a.cols() << " columns by " << matrix_a.rows()
				<< " rows, Matrix b: " << matrix_b.cols() << " columns by " << matrix_b.rows() << " rows."
				<< std::endl;
		return false;
	}
	for (Eigen::Index index = 0; index < matrix_a.size(); index++) {
		if (!(std::abs(matrix_a(index) - (matrix_b(index)) < tolerance))) {
			ldiv_t division_result = div(index, matrix_a.cols());
			long x = division_result.quot;
			long y = division_result.rem;
			std::cout << "Matrix entries are not within tolerance threshold of each other. First mismatch at row " << y
					<< ", column " << x << ". " << "Values: " << matrix_a(index) << " vs. " << matrix_b(index)
					<< ", difference: " << matrix_a(index) - matrix_b(index) << std::endl;
			return false;
		}
	}
	return true;
}

//TODO Tensor almost equal
template<typename TEigenContainer>
bool almost_equal_verbose(TEigenContainer container_a, TEigenContainer container_b,
		const std::function<bool(TEigenContainer container_a, TEigenContainer container_b)>& compare_dimensions,
		const std::function<bool(typename TEigenContainer::Scalar element_a,
				typename TEigenContainer::Scalar element_b, double tolerance)>& compare_elements,
		const std::function<void(TEigenContainer container_a, TEigenContainer container_b, Eigen::Index index)>& print_local_error,
		double tolerance
		) {
	compare_dimensions(container_a, container_b);
	for (Eigen::Index index = 0; index < container_a.size(); index++) {
		if (!compare_elements(container_a(index), container_b(index))) {
			print_local_error(container_a, container_b, index);
			return false;
		}
	}
	return true;
}

typedef Eigen::Tensor<float, 3> ten3;

//TODO: break up into generic, regular, and nested
template<typename TTensor>
bool tensor_almost_equal_verbose(TTensor container_a, TTensor container_b) {
	return almost_equal_verbose<TTensor>(container_a, container_b,
			[](TTensor container_a, TTensor container_b)-> bool {
				for (int i_dim = 0; i_dim < TTensor::NumDimensions; i_dim++){
					if(container_a.dimension(i_dim) != container_b.dimension(i_dim)){
						std::cout << "Tensor dimension " << i_dim << " (0-based) doesn't match. Tensor a dimension: "
								<< container_a.dimension(i_dim) << ". Corresponding tensor b dimension: " <<
								container_b.dimension(i_dim) << std::endl;
						return false;
					}
				}
				return true;
			},
			[](typename TTensor::Scalar element_a, typename TTensor::Scalar element_b, double tolerance)-> bool {
				return static_cast<double>(std::abs(element_a - element_b)) < tolerance;
			},
			[](TTensor container_a, TTensor container_b, Eigen::Index index) -> void {

			}

	);
}

} //namespace math
