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

#include "assessment.hpp"

namespace math {
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
}//namespace math


