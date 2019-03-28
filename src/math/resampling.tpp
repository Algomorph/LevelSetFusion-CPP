/*
 * resampling.tpp
 *
 *  Created on: Mar 28, 2019
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
#include "resampling.hpp"

namespace math {

template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> upsampleX2(
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& matrix) {
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> upsampled(matrix.rows() * 2,
			matrix.cols() * 2);
	for (int dest_col = 0, source_col = 0; source_col < matrix.cols(); dest_col += 2, source_col++) {
		for (int dest_row = 0, source_row = 0; source_row < matrix.rows(); dest_row += 2, source_row++) {
			Scalar value = matrix(source_row, source_col);
			upsampled(dest_row, dest_col) = value;
			upsampled(dest_row + 1, dest_col) = value;
			upsampled(dest_row, dest_col + 1) = value;
			upsampled(dest_row + 1, dest_col + 1) = value;
		}
	}
	return upsampled;
}

template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> upsampleX2(
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& matrix) {
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> upsampled(matrix.rows() * 2,
			matrix.cols() * 2);
	for (int dest_row = 0, source_row = 0; source_row < matrix.rows(); dest_row += 2, source_row++) {
		for (int dest_col = 0, source_col = 0; source_col < matrix.cols(); dest_col += 2, source_col++) {
			Scalar value = matrix(source_row, source_col);
			upsampled(dest_row, dest_col) = value;
			upsampled(dest_row + 1, dest_col) = value;
			upsampled(dest_row, dest_col + 1) = value;
			upsampled(dest_row + 1, dest_col + 1) = value;
		}
	}
	return upsampled;
}

} // namespace math
