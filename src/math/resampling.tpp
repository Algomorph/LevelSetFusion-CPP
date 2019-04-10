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

//stdlib
#include <cassert>

//local
#include "resampling.hpp"
#include "checks.hpp"
#include "../error_handling/throw_assert.hpp"
#include "padding.hpp"

namespace eig = Eigen;
namespace math {

// region ================================ GENERIC UPSAMPLING ==========================================================
template<typename ContainerType, typename Scalar>
static inline ContainerType upsampleX2_aux(const ContainerType& field, UpsamplingStrategy upsampling_strategy) {
	switch (upsampling_strategy) {
	case UpsamplingStrategy::NEAREST:
		return upsampleX2_nearest(field);
	case UpsamplingStrategy::LINEAR:
		return upsampleX2_linear(field);
	default:
		throw_assert(false, "Unknown UpsamplingStrategy")
		;
		return ContainerType();
	}
}

template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> upsampleX2(
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& field,
		UpsamplingStrategy upsampling_strategy) {
	return upsampleX2_aux<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, Scalar>(field,
			upsampling_strategy);
}

template<typename Scalar>
Eigen::Tensor<Scalar, 3, Eigen::ColMajor> upsampleX2(
		const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& field,
		UpsamplingStrategy upsampling_strategy) {
	return upsampleX2_aux<Eigen::Tensor<Scalar, 3, Eigen::ColMajor>, Scalar>(field, upsampling_strategy);
}

// endregion
// region ================================ NEAREST UPSAMPLING ==========================================================
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

//TODO: write tests for this
template<typename Scalar>
Eigen::Tensor<Scalar, 3, Eigen::ColMajor> upsampleX2_nearest(
		const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& field) {
	Eigen::Tensor<Scalar, 3, Eigen::ColMajor> upsampled(field.dimension(0) * 2, field.dimension(1) * 2,
			field.dimension(2) * 2);
#pragma omp parallel for
	for (int z_source = 0; z_source < field.dimension(2); z_source++) {
		int z_dest = z_source * 2;
		for (int y_dest = 0, y_source = 0; y_source < field.dimension(1); y_dest += 2, y_source++) {
			for (int x_dest = 0, x_source = 0; x_source < field.dimension(0); x_dest += 2, x_source++) {
				Scalar value = field(x_source, y_source, z_source);
				upsampled(x_dest, y_dest, z_dest) = value;
				upsampled(x_dest + 1, y_dest, z_dest) = value;
				upsampled(x_dest, y_dest + 1, z_dest) = value;
				upsampled(x_dest + 1, y_dest + 1, z_dest) = value;

				upsampled(x_dest, y_dest, z_dest + 1) = value;
				upsampled(x_dest + 1, y_dest, z_dest + 1) = value;
				upsampled(x_dest, y_dest + 1, z_dest + 1) = value;
				upsampled(x_dest + 1, y_dest + 1, z_dest + 1) = value;
			}
		}
	}
	return upsampled;
}
//endregion
//region ====================================== LINEAR UPSAMPLING ======================================================
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
Eigen::Tensor<Scalar, 3, Eigen::ColMajor> upsampleX2_linear(
		const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& field) {
	typedef Eigen::Tensor<Scalar, 3, Eigen::ColMajor> TensorType;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixType;

	int size_x = field.dimension(0), size_y = field.dimension(1), size_z = field.dimension(2);
	int usize_x = size_x * 2, usize_y = size_y * 2, usize_z = size_z * 2;
	int ulx = usize_x - 1, uly = usize_y - 1, ulz = usize_z - 1;

	TensorType upsampled(usize_x, usize_y, usize_z);

	//the sides can be processed as matrices.

	typedef eig::array<long, 3> Arrl3;
	typedef eig::array<long, 2> Arrl2;

	Arrl3 extents_source[] = {
			Arrl3 { 1, size_y, size_z },
			Arrl3 { size_x, 1, size_z },
			Arrl3 { size_x, size_y, 1 }
	};
	Arrl2 mat_extents_source[] {
			Arrl2 { size_y, size_z },
			Arrl2 { size_x, size_z },
			Arrl2 { size_x, size_y }
	};
	Arrl3 extents_target[] = {
			Arrl3 { 1, usize_y, usize_z },
			Arrl3 { usize_x, 1, usize_z },
			Arrl3 { usize_x, usize_y, 1 }
	};
	struct FaceExtentAndOffset {
		Arrl3 offset_source;
		Arrl3 extent_source;
		Arrl2 mat_extent_source;
		Arrl3 offset_target;
		Arrl3 extent_target;
	};
	FaceExtentAndOffset face_extents_and_offsets[] = { //@formatter:off
			{Arrl3{0,0,0},        extents_source[0], mat_extents_source[0], Arrl3{0,0,  0}, extents_target[0]},  //near x
			{Arrl3{size_x-1,0,0}, extents_source[0], mat_extents_source[0], Arrl3{ulx,0,0}, extents_target[0]},  //far x
			{Arrl3{0,0,0},        extents_source[1], mat_extents_source[1], Arrl3{0,0,  0}, extents_target[1]},  //near y
			{Arrl3{0,size_y-1,0}, extents_source[1], mat_extents_source[1], Arrl3{0,uly,0}, extents_target[1]},  //far y
			{Arrl3{0,0,0},        extents_source[2], mat_extents_source[2], Arrl3{0,0,  0}, extents_target[2]},  //near z
			{Arrl3{0,0,size_z-1}, extents_source[2], mat_extents_source[2], Arrl3{0,0,ulz}, extents_target[2]}   //far z
	};//@formatter:on

	for (const FaceExtentAndOffset& feao : face_extents_and_offsets) {
		TensorType slice_source = field.slice(feao.offset_source, feao.extent_source);
		MatrixType slice_matrix_source = eig::Map<MatrixType>(
				slice_source.data(),
				feao.mat_extent_source[0],
				feao.mat_extent_source[1]
				);
		MatrixType slice_matrix_upsampled = upsampleX2_linear(slice_matrix_source);
		eig::TensorMap<TensorType, eig::Aligned> slice_upsampled(slice_matrix_upsampled.data(), feao.extent_target);
		upsampled.slice(feao.offset_target, feao.extent_target) = slice_upsampled;
	}

#pragma omp parallel for
	for (int z_source = 0; z_source < size_z - 1; z_source++) {
		int z_dest = z_source * 2 + 1;
		for (int y_dest = 1, y_source = 0; y_source < size_y - 1; y_dest += 2, y_source++) {
			for (int x_dest = 1, x_source = 0; x_source < size_x - 1; x_dest += 2, x_source++) {
				//@formatting:off
				Scalar v000 = field(x_source, y_source, z_source);
				Scalar v100 = field(x_source + 1, y_source, z_source);
				Scalar v010 = field(x_source, y_source + 1, z_source);
				Scalar v110 = field(x_source + 1, y_source + 1, z_source);
				Scalar v001 = field(x_source, y_source, z_source + 1);
				Scalar v101 = field(x_source + 1, y_source, z_source + 1);
				Scalar v011 = field(x_source, y_source + 1, z_source + 1);
				Scalar v111 = field(x_source + 1, y_source + 1, z_source + 1); //@formatting:on
				//interpolate along x dimension
				Scalar xv000 = 0.75f * v000 + 0.25f * v100;
				Scalar xv100 = 0.25f * v000 + 0.75f * v100;
				Scalar xv010 = 0.75f * v010 + 0.25f * v110;
				Scalar xv110 = 0.25f * v010 + 0.75f * v110;
				Scalar xv001 = 0.75f * v001 + 0.25f * v101;
				Scalar xv101 = 0.25f * v001 + 0.75f * v101;
				Scalar xv011 = 0.75f * v011 + 0.25f * v111;
				Scalar xv111 = 0.25f * v011 + 0.75f * v111;
				//interpolate along y dimension
				Scalar yv000 = 0.75f * xv000 + 0.25f * xv010;
				Scalar yv010 = 0.25f * xv000 + 0.75f * xv010;
				Scalar yv100 = 0.75f * xv100 + 0.25f * xv110;
				Scalar yv110 = 0.25f * xv100 + 0.75f * xv110;
				Scalar yv001 = 0.75f * xv001 + 0.25f * xv011;
				Scalar yv011 = 0.25f * xv001 + 0.75f * xv011;
				Scalar yv101 = 0.75f * xv101 + 0.25f * xv111;
				Scalar yv111 = 0.25f * xv101 + 0.75f * xv111;
				//interpolate along z dimension
				upsampled(x_dest, y_dest, z_dest) = 0.75f * yv000 + 0.25f * yv001;
				upsampled(x_dest + 1, y_dest, z_dest) = 0.75f * yv100 + 0.25f * yv101;
				upsampled(x_dest, y_dest + 1, z_dest) = 0.75f * yv010 + 0.25f * yv011;
				upsampled(x_dest + 1, y_dest + 1, z_dest) = 0.75f * yv110 + 0.25f * yv111;
				upsampled(x_dest, y_dest, z_dest + 1) = 0.25f * yv000 + 0.75f * yv001;
				upsampled(x_dest + 1, y_dest, z_dest + 1) = 0.25f * yv100 + 0.75f * yv101;
				upsampled(x_dest, y_dest + 1, z_dest + 1) = 0.25f * yv010 + 0.75f * yv011;
				upsampled(x_dest + 1, y_dest + 1, z_dest + 1) = 0.25f * yv110 + 0.75f * yv111;
			}
		}
	}
	return upsampled;
}
//endregion
//region ===================================== GENERIC DOWNSAMPLING ====================================================
template<typename ContainerType, typename Scalar>
static inline ContainerType downsampleX2_aux(
		const ContainerType& field,
		DownsamplingStrategy downsampling_strategy) {
	switch (downsampling_strategy) {
	case DownsamplingStrategy::AVERAGE:
		return downsampleX2_average(field);
	case DownsamplingStrategy::LINEAR:
		return downsampleX2_linear(field);
	default:
		throw_assert(false, "Unknown UpsamplingStrategy")
		;
		return ContainerType();
	}
}

template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> downsampleX2(
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& field,
		DownsamplingStrategy downsampling_strategy) {
	return downsampleX2_aux<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, Scalar>(field,
			downsampling_strategy);
}

template<typename Scalar>
Eigen::Tensor<Scalar, 3, Eigen::ColMajor> downsampleX2(
		const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& field,
		DownsamplingStrategy downsampling_strategy) {
	return downsampleX2_aux<Eigen::Tensor<Scalar, 3, Eigen::ColMajor>, Scalar>(field, downsampling_strategy);
}
//endregion
//region ============================== DOWNSAMPLING USING AVERAGE STRATEGY ============================================
template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> downsampleX2_average(
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& field) {

	throw_assert((math::is_power_of_two(field.rows()) && math::is_power_of_two(field.cols())),
			"The argument 'field' must have a power of two for each dimension.");

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
Eigen::Tensor<Scalar, 3, Eigen::ColMajor> downsampleX2_average(
		const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& field) {
	eig::Tensor<Scalar, 3> downsampled(
			field.dimension(0) / 2,
			field.dimension(1) / 2,
			field.dimension(2) / 2
					);
	//average each square of 4 cells into one
	for (int z_target = 0, z_source = 0;
			z_target < downsampled.dimension(2);
			z_target++, z_source += 2) {
		for (eig::Index y_target = 0, y_source = 0;
				y_target < downsampled.dimension(1);
				y_target++, y_source += 2) {
			for (eig::Index x_target = 0, x_source = 0;
					x_target < downsampled.dimension(0);
					x_target++, x_source += 2) {
				/* @formatter:off*/
				downsampled(x_target, y_target, z_target) = (
						field(x_source, y_source, z_source) +
						field(x_source + 1, y_source, z_source) +
						field(x_source, y_source + 1, z_source) +
						field(x_source + 1, y_source + 1, z_source) +
						field(x_source, y_source, z_source + 1) +
						field(x_source + 1, y_source, z_source + 1) +
						field(x_source, y_source + 1, z_source + 1)+
						field(x_source + 1, y_source + 1, z_source + 1)
				) / 8.0f;/* @formatter:on */
			}
		}
	}
	return downsampled;
}

//endregion
//region ============================== DOWNSAMPLING USING LINEAR STRATEGY =============================================
template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> downsampleX2_linear(
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& field) {
	throw_assert((field.rows() % 2 == 0 && field.cols() % 2 == 0 && field.rows() > 2 && field.cols() > 2),
			"Each dimension of the argument 'field' must be divisible by 2 and greater than 2.");

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
#pragma omp parallel for
	for (eig::Index target_col = 1; target_col < downsampled.cols() - 1;
			target_col++) {
		eig::Index source_col = target_col * 2;
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
#pragma omp parallel for
	for (eig::Index target_row = 1; target_row < downsampled.rows() - 1; target_row++) {
		eig::Index source_row = target_row * 2;
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

template<typename Scalar>
Eigen::Tensor<Scalar, 3, Eigen::ColMajor> downsampleX2_linear(
		const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& field) {

	throw_assert(field.dimension(0) % 2 == 0 && field.dimension(1) % 2 == 0 && field.dimension(2) % 2 == 0 &&
			field.dimension(0) > 2 && field.dimension(1) > 2 && field.dimension(2) > 2,
			"Each dimension of the argument 'field' must be divisible by 2 and greater than 2.");
	typedef Eigen::Tensor<Scalar, 3, Eigen::ColMajor> TensorType;
	int size_x = field.dimension(0), size_y = field.dimension(1), size_z = field.dimension(2);
	int dsize_x = size_x / 2, dsize_y = size_y / 2, dsize_z = size_z / 2;
	TensorType downsampled(dsize_x, dsize_y, dsize_z);
	//multiplication now and division later increases precision
	const float c0 = 0.052734375f * 4.0f;
	const float c1 = 0.017578125f * 4.0f;
	const float c2 = 0.005859375f * 4.0f;
	const float c3 = 0.001953125f * 4.0f;

	TensorType padded = math::pad_replicate(field, 1);


#pragma omp parallel for
	for (int x_target = 0; x_target < dsize_x; x_target++) {
		int x_source = x_target * 2 + 1;
		for (int y_target = 0, y_source = 1; y_target < dsize_y; y_target++, y_source += 2) {
			for (int z_target = 0, z_source = 1; z_target < dsize_z; z_target++, z_source += 2) {
				downsampled(x_target, y_target, z_target) = //@formatter:off
					   (c0 * (padded(x_source + 0, y_source + 0, z_source + 0) +
							  padded(x_source + 1, y_source + 0, z_source + 0) +
							  padded(x_source + 0, y_source + 1, z_source + 0) +
							  padded(x_source + 1, y_source + 1, z_source + 0) +
							  padded(x_source + 0, y_source + 0, z_source + 1) +
							  padded(x_source + 1, y_source + 0, z_source + 1) +
							  padded(x_source + 0, y_source + 1, z_source + 1) +
							  padded(x_source + 1, y_source + 1, z_source + 1)) +

					    c1 * (padded(x_source - 1, y_source + 0, z_source + 0) +
							  padded(x_source + 0, y_source - 1, z_source + 0) +
							  padded(x_source + 0, y_source + 0, z_source - 1) +

							  padded(x_source + 2, y_source + 0, z_source + 0) +
							  padded(x_source + 1, y_source - 1, z_source + 0) +
							  padded(x_source + 1, y_source + 0, z_source - 1) +

							  padded(x_source - 1, y_source + 1, z_source + 0) +
							  padded(x_source + 0, y_source + 2, z_source + 0) +
							  padded(x_source + 0, y_source + 1, z_source - 1) +

							  padded(x_source + 2, y_source + 1, z_source + 0) +
							  padded(x_source + 1, y_source + 2, z_source + 0) +
							  padded(x_source + 1, y_source + 1, z_source - 1) +

							  padded(x_source - 1, y_source + 0, z_source + 1) +
							  padded(x_source + 0, y_source - 1, z_source + 1) +
							  padded(x_source + 0, y_source + 0, z_source + 2) +

							  padded(x_source + 2, y_source + 0, z_source + 1) +
							  padded(x_source + 1, y_source - 1, z_source + 1) +
							  padded(x_source + 1, y_source + 0, z_source + 2) +

							  padded(x_source - 1, y_source + 1, z_source + 1) +
							  padded(x_source + 0, y_source + 2, z_source + 1) +
							  padded(x_source + 0, y_source + 1, z_source + 2) +

							  padded(x_source + 2, y_source + 1, z_source + 1) +
							  padded(x_source + 1, y_source + 2, z_source + 1) +
							  padded(x_source + 1, y_source + 1, z_source + 2)) +

						c2 * (padded(x_source - 1, y_source - 1, z_source + 0) +
							  padded(x_source + 0, y_source - 1, z_source - 1) +
							  padded(x_source - 1, y_source + 0, z_source - 1) +

							  padded(x_source + 2, y_source - 1, z_source + 0) +
							  padded(x_source + 1, y_source - 1, z_source - 1) +
							  padded(x_source + 2, y_source + 0, z_source - 1) +

							  padded(x_source - 1, y_source + 2, z_source + 0) +
							  padded(x_source + 0, y_source + 2, z_source - 1) +
							  padded(x_source - 1, y_source + 1, z_source - 1) +

							  padded(x_source + 2, y_source + 2, z_source + 0) +
							  padded(x_source + 1, y_source + 2, z_source - 1) +
							  padded(x_source + 2, y_source + 1, z_source - 1) +

							  padded(x_source - 1, y_source - 1, z_source + 1) +
							  padded(x_source + 0, y_source - 1, z_source + 2) +
							  padded(x_source - 1, y_source + 0, z_source + 2) +

							  padded(x_source + 2, y_source - 1, z_source + 1) +
							  padded(x_source + 1, y_source - 1, z_source + 2) +
							  padded(x_source + 2, y_source + 0, z_source + 2) +

							  padded(x_source - 1, y_source + 2, z_source + 1) +
							  padded(x_source + 0, y_source + 2, z_source + 2) +
							  padded(x_source - 1, y_source + 1, z_source + 2) +

							  padded(x_source + 2, y_source + 2, z_source + 1) +
							  padded(x_source + 1, y_source + 2, z_source + 2) +
							  padded(x_source + 2, y_source + 1, z_source + 2)) +

					    c3 * (padded(x_source - 1, y_source - 1, z_source - 1) +
							  padded(x_source + 2, y_source - 1, z_source - 1) +
							  padded(x_source - 1, y_source + 2, z_source - 1) +
							  padded(x_source + 2, y_source + 2, z_source - 1) +
							  padded(x_source - 1, y_source - 1, z_source + 2) +
							  padded(x_source + 2, y_source - 1, z_source + 2) +
							  padded(x_source - 1, y_source + 2, z_source + 2) +
							  padded(x_source + 2, y_source + 2, z_source + 2))) * 0.25f;

				;//@formatter:on
			}
		}
	}
	return downsampled;
}

} // namespace math
