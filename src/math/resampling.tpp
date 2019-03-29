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

namespace eig = Eigen;
namespace math {

template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> upsampleX2(
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& field) {
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> upsampled(field.rows() * 2,
			field.cols() * 2);
	for (int dest_col = 0, source_col = 0; source_col < field.cols(); dest_col += 2, source_col++) {
		for (int dest_row = 0, source_row = 0; source_row < field.rows(); dest_row += 2, source_row++) {
			Scalar value = field(source_row, source_col);
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
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& field) {
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> upsampled(field.rows() * 2,
			field.cols() * 2);
	for (int dest_row = 0, source_row = 0; source_row < field.rows(); dest_row += 2, source_row++) {
		for (int dest_col = 0, source_col = 0; source_col < field.cols(); dest_col += 2, source_col++) {
			Scalar value = field(source_row, source_col);
			upsampled(dest_row, dest_col) = value;
			upsampled(dest_row + 1, dest_col) = value;
			upsampled(dest_row, dest_col + 1) = value;
			upsampled(dest_row + 1, dest_col + 1) = value;
		}
	}
	return upsampled;
}

template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> upsampleX2_bilinear(
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& field) {

	eig::Index upsampled_cols = field.cols() * 2;
	eig::Index upsampled_rows = field.rows() * 2;

	eig::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> upsampled(upsampled_rows, upsampled_cols);
	Scalar current_value_0, current_value_1, prev_value_0, prev_value_1;

	// process first and last rows of destination (including corners)
	int source_col = 0;
	int source_row_0 = 0;
	int source_row_1 = field.rows() - 1;
	int dest_col = 0;
	int dest_row_0 = 0;
	int dest_row_1 = upsampled_rows - 1;

	Scalar corner_ul = field(source_row_0, source_col);
	Scalar corner_bl = field(source_row_1, source_col);
	upsampled(dest_row_0, dest_col) = corner_ul; //corner x=0, y=0
	upsampled(dest_row_1, dest_col) = corner_bl; //corner x=0, y=height-1

	prev_value_0 = corner_ul;
	prev_value_1 = corner_bl;
	for (source_col = 1, dest_col = 1; source_col < field.cols(); source_col++, dest_col += 2) {
		current_value_0 = field(source_row_0, source_col);
		current_value_1 = field(source_row_1, source_col);
		upsampled(dest_row_0, dest_col) = 0.75f * prev_value_0 + 0.25f * current_value_0;
		upsampled(dest_row_0, dest_col + 1) = 0.25f * prev_value_0 + 0.75f * current_value_0;
		upsampled(dest_row_1, dest_col) = 0.75f * prev_value_1 + 0.25f * current_value_1;
		upsampled(dest_row_1, dest_col + 1) = 0.25f * prev_value_1 + 0.75f * current_value_1;
		prev_value_0 = current_value_0;
		prev_value_1 = current_value_1;
	}
	dest_col = upsampled_cols - 1;
	upsampled(dest_row_0, dest_col) = prev_value_0; //corner x=width-1, y=0
	upsampled(dest_row_1, dest_col) = prev_value_1; //corner x=width-1, y=height-1

	// process first and last columns of destination (excluding corners)
	int source_row = 0;
	int source_col_0 = 0;
	int source_col_1 = field.cols() - 1;
	int dest_row = 0;
	int dest_col_0 = 0;
	int dest_col_1 = upsampled_cols - 1;

	Scalar corner_ur = field(source_row, source_col_1);

	prev_value_0 = corner_ul;
	prev_value_1 = corner_ur;
	for (source_row = 1, dest_row = 1; source_row < field.rows(); source_row++, dest_row += 2) {
		current_value_0 = field(source_row, source_col_0);
		current_value_1 = field(source_row, source_col_1);
		upsampled(dest_row, dest_col_0) = 0.75f * prev_value_0 + 0.25f * current_value_0;
		upsampled(dest_row + 1, dest_col_0) = 0.25f * prev_value_0 + 0.75f * current_value_0;
		upsampled(dest_row, dest_col_1) = 0.75f * prev_value_1 + 0.25f * current_value_1;
		upsampled(dest_row + 1, dest_col_1) = 0.25f * prev_value_1 + 0.75f * current_value_1;
		prev_value_0 = current_value_0;
		prev_value_1 = current_value_1;
	}

	// process the section of destination excluding the boundaries ("midsection")
#pragma omp parallel for
	for (int source_col = 0; source_col < field.cols() - 1; source_col++) {
		int dest_col = 1 + 2 * source_col;
		Scalar value00 = field(0, source_col);
		Scalar value01 = field(0, source_col + 1);
		for (int source_row = 1, dest_row = 1; source_row < field.rows(); source_row++, dest_row += 2) {
			Scalar value10 = field(source_row, source_col);
			Scalar value11 = field(source_row, source_col + 1);
			//@formatter:off
			upsampled(dest_row, dest_col)         = 0.5625f*value00 + 0.1875f*value01 + 0.1875f*value10 + 0.0625f*value11;
			upsampled(dest_row, dest_col + 1)     = 0.1875f*value00 + 0.5625f*value01 + 0.0625f*value10 + 0.1875f*value11;
			upsampled(dest_row + 1, dest_col)     = 0.1875f*value00 + 0.0625f*value01 + 0.5625f*value10 + 0.1875f*value11;
			upsampled(dest_row + 1, dest_col + 1) = 0.0625f*value00 + 0.1875f*value01 + 0.1875f*value10 + 0.5625f*value11;
			//@formatter:on
			value00 = value10;
			value01 = value11;
		}
	}

	return upsampled;
}

} // namespace math
