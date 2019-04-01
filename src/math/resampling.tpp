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
#include "checks.hpp"

namespace eig = Eigen;
namespace math {

template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> upsampleX2_nearest(
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
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> upsampleX2_nearest(
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
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> upsampleX2_linear(
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
			upsampled(dest_row, dest_col) = 0.5625f * value00 + 0.1875f * value01 + 0.1875f * value10
					+ 0.0625f * value11;
			upsampled(dest_row, dest_col + 1) = 0.1875f * value00 + 0.5625f * value01 + 0.0625f * value10
					+ 0.1875f * value11;
			upsampled(dest_row + 1, dest_col) = 0.1875f * value00 + 0.0625f * value01 + 0.5625f * value10
					+ 0.1875f * value11;
			upsampled(dest_row + 1, dest_col + 1) = 0.0625f * value00 + 0.1875f * value01 + 0.1875f * value10
					+ 0.5625f * value11;
																																																						//@formatter:on
			value00 = value10;
			value01 = value11;
		}
	}

	return upsampled;
}

template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> downsampleX2_average(
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& field) {
#ifndef NDEBUG
	eigen_assert((math::is_power_of_two(field.rows()) && math::is_power_of_two(field.cols()))
			&& "The argument 'field' must have a power of two for each dimension.");
#endif

	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> downsampled(field.rows() / 2,
			field.cols() / 2);
	//average each square of 4 cells into one
#pragma omp parallel for
	for (eig::Index target_col = 0; target_col < downsampled.cols(); target_col++) {
		eig::Index source_col = target_col * 2;
		for (eig::Index i_downsampled_row = 0, source_row = 0;
				i_downsampled_row < downsampled.rows();
				i_downsampled_row++, source_row += 2) {
			downsampled(i_downsampled_row, target_col) = ( //@formatter:off
					field(source_row, source_col) +
					field(source_row, source_col + 1) +
					field(source_row + 1, source_col) +
					field(source_row + 1, source_col + 1) //@formatter:on
			) / 4.0f;
		}
	}
	return downsampled;
}

template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> downsampleX2_linear(
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& field) {
#ifndef NDEBUG
	eigen_assert((field.rows() % 2 == 0 && field.cols() % 2 == 0 &&
					field.rows() > 2 && field.cols() > 2)
			&& "Each dimension of the argument 'field' must be divisible by 2 and greater than 2.");
#endif
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> downsampled(field.rows() / 2,
			field.cols() / 2);

	const float coeff0 = 0.140625f;
	const float coeff1 = 0.046875f;
	const float coeff2 = 0.015625f;

	Eigen::Index lc = field.cols() - 1, lr = field.rows() - 1;
	Eigen::Index dlc = downsampled.cols() - 1, dlr = downsampled.rows() - 1;
	//upper-left //@formatter:off
	downsampled(0, 0) =
			coeff0 * (field(0, 0)       + field(1, 0)           + field(0, 1)           + field(1, 1)) +
			coeff1 * (field(0, 0)       + field(0, 0)           + field(0, 1)           + field(1, 0) +
					  field(0, 2)       + field(1, 2)           + field(2, 1)           + field(2, 0)) +
			coeff2 * (field(0, 0)       + field(0, 2)           + field(2, 0)           + field(2, 2));
	//upper-right
	downsampled(0, dlc) =
			coeff0 * (field(0, lc)      + field(1, lc)          + field(0, lc - 1)      + field(1, lc - 1)) +
			coeff1 * (field(0, lc)      + field(0, lc)          + field(0, lc - 1)      + field(1, lc) +
					  field(0, lc - 2)  + field(1, lc - 2)      + field(2, lc - 1)      + field(2, lc)) +
			coeff2 * (field(0, lc)      + field(0, lc - 2)      + field(2, lc)          + field(2, lc - 2));
	//lower-left
	downsampled(dlr, 0) =
			coeff0 * (field(lr, 0)      + field(lr - 1, 0)      + field(lr, 1)          + field(lr - 1, 1)) +
			coeff1 * (field(lr, 0)      + field(lr, 0)          + field(lr, 1)          + field(lr - 1, 0) +
					  field(lr, 2)      + field(lr - 1, 2)      + field(lr - 2, 1)      + field(lr - 2, 0)) +
			coeff2 * (field(lr, 0)      + field(lr, 2)          + field(lr - 2, 0)      + field(lr - 2, 2));
	//lower-right
	downsampled(dlr, dlc) =
			coeff0 * (field(lr, lc)     + field(lr - 1, lc)     + field(lr, lc - 1)     + field(lr - 1, lc - 1)) +
			coeff1 * (field(lr, lc)     + field(lr, lc)         + field(lr, lc - 1)     + field(lr - 1, lc) +
					  field(lr, lc - 2) + field(lr - 1, lc - 2) + field(lr - 2, lc - 1) + field(lr - 2, lc)) +
			coeff2 * (field(lr, lc)     + field(lr, lc - 2)     + field(lr - 2, lc)     + field(lr - 2, lc - 2)); //@formatter:on
	//upper row, lower row
//#pragma omp parallel for //TODO -- parallelize properly, omp for loop cannot have extra loop vars
	for (eig::Index target_col = 1, source_col = 2; target_col < downsampled.cols() - 1;
			target_col++, source_col += 2) {
		downsampled(0, target_col) = //@formatter:off
			coeff0 * (field(0, source_col)     + field(0, source_col + 1) +
					  field(1, source_col)     + field(1, source_col + 1)) +

			coeff1 * (field(0, source_col - 1) + field(0, source_col) +
					  field(0, source_col + 1) + field(0, source_col + 2) +
					  field(1, source_col - 1) + field(2, source_col) +
					  field(2, source_col + 1) + field(1, source_col + 2)) +

			coeff2 * (field(0, source_col - 1) + field(0, source_col + 2) +
					  field(2, source_col - 1) + field(2, source_col + 2))
		;
		downsampled(dlr, target_col) =
			coeff0 * (field(lr, source_col)         + field(lr, source_col + 1) +
					  field(lr - 1, source_col)     + field(lr - 1, source_col + 1)) +

			coeff1 * (field(lr, source_col - 1)     + field(lr, source_col) +
					  field(lr, source_col + 1)     + field(lr, source_col + 2) +
					  field(lr - 1, source_col - 1) + field(lr - 2, source_col) +
					  field(lr - 2, source_col + 1) + field(lr - 1, source_col + 2)) +

			coeff2 * (field(lr, source_col - 1)     + field(lr, source_col + 2) +
					  field(lr - 2, source_col - 1) + field(lr - 2, source_col + 2))
		; //@formatter:on
	}

	//left column, right column
//#pragma omp parallel for //TODO -- parallelize properly, omp for loop cannot have extra loop vars
	for (eig::Index target_row = 1, source_row = 2; target_row < downsampled.rows() - 1;
			target_row++, source_row += 2) {
		downsampled(target_row, 0) = //@formatter:off
			coeff0 * (field(source_row, 0) + field(source_row + 1, 0) +
					  field(source_row, 1) + field(source_row + 1, 1)) +

			coeff1 * (field(source_row - 1, 0) + field(source_row, 0) +
					  field(source_row + 1, 0) + field(source_row + 2, 0) +
					  field(source_row - 1, 1) + field(source_row, 2) +
					  field(source_row + 1, 2) + field(source_row + 2, 1)) +

			coeff2 * (field(source_row - 1, 0) + field(source_row + 2, 0) +
					  field(source_row - 1, 2) + field(source_row + 2, 2))
		; //@formatter:on
		downsampled(target_row, dlc) = //@formatter:off
			coeff0 * (field(source_row, lc - 0) + field(source_row + 1, lc - 0) +
					  field(source_row, lc - 1) + field(source_row + 1, lc - 1)) +

			coeff1 * (field(source_row - 1, lc - 0) + field(source_row, lc - 0) +
					  field(source_row + 1, lc - 0) + field(source_row + 2, lc - 0) +
					  field(source_row - 1, lc - 1) + field(source_row, lc - 2) +
					  field(source_row + 1, lc - 2) + field(source_row + 2, lc - 1)) +

			coeff2 * (field(source_row - 1, lc - 0) + field(source_row + 2, lc - 0) +
					  field(source_row - 1, lc - 2) + field(source_row + 2, lc - 2))
		; //@formatter:on
	}

#pragma omp parallel for
	for (eig::Index target_col = 1; target_col < downsampled.cols() - 1; target_col++) {
		eig::Index source_col = target_col * 2;
		for (eig::Index target_row = 1, source_row = 2; target_row < downsampled.rows() - 1;
				target_row++, source_row += 2) {
			downsampled(target_row, target_col) = //@formatter:off
				coeff0 * (field(source_row, source_col) 		+ field(source_row, source_col + 1) +
						  field(source_row + 1, source_col) 	+ field(source_row + 1, source_col + 1)) +

				coeff1 * (field(source_row - 1, source_col) 	+ field(source_row, source_col - 1) +
						  field(source_row - 1, source_col + 1) + field(source_row, source_col + 2) +
						  field(source_row + 2, source_col) 	+ field(source_row + 1, source_col - 1) +
						  field(source_row + 2, source_col + 1) + field(source_row + 1, source_col + 2)) +

				coeff2 * (field(source_row - 1, source_col - 1) + field(source_row - 1, source_col + 2) +
						  field(source_row + 2, source_col - 1) + field(source_row + 2, source_col + 2))
			; //@formatter:on
		}
	}
	return downsampled;
}

} // namespace math
