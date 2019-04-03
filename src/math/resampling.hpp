/*
 * resampling.hpp
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

//libraries
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace math {

/**
 * Different strategies for upsampling a discrete field
 */
enum class UpsamplingStrategy {
	NEAREST = 0, ///@see upsampleX2_nearest for details
	LINEAR = 1 ///@see upsampleX2_linear for details
};

/**
 * Different strategies for downsampling a discrete field
 */
enum class DownsamplingStrategy {
	AVERAGE = 0, ///@see downsampleX2_average for details
	LINEAR = 1 ///@see downsampleX2_linear for details
};

/**
 * Upsample the field such that the output is a field 2X larger than the original in each dimension.
 * @param field input field
 * @param upsampling_strategy -- which upsampling strategy to use.
 * @return upsampled field
 */
template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> upsampleX2(
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& field,
		UpsamplingStrategy upsampling_strategy = UpsamplingStrategy::NEAREST);
/**
 * Upsample the field such that the output is a field 2X larger than the original in each dimension.
 * @overload
 */
template<typename Scalar>
Eigen::Tensor<Scalar, 3, Eigen::ColMajor> upsampleX2(
		const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& field,
		UpsamplingStrategy upsampling_strategy = UpsamplingStrategy::NEAREST);

/**
 * Upsample the field such that the output is a field 2X larger than the original in each dimension.
 * This procedure uses a simple box filter / no interpolation, i.e. simply copies the value to it's immediate "children"
 * in the upsampled version.
 * Conceptual example:
 * ⎡1  2⎤
 * ⎣3  4⎦
 * yields
 * ⎡1  1  2  2⎤
 * ⎢1  1  2  2⎥
 * ⎢3  3  4  4⎥
 * ⎣3  3  4  4⎦
 * @param field input field
 * @return upsampled field
 */
template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> upsampleX2_nearest(
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& field);

/**
 * Upsample the field such that the output is a field 2X larger than the original in each dimension using the
 * nearest-neighbor (NEAREST) strategy.
 * @overload
 */
template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> upsampleX2_nearest(
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& field);

/**
 * Upsample the field such that the output is a field 2X larger than the original in each dimension.
 * @overload
 */
//TODO
template<typename Scalar>
Eigen::Tensor<Scalar, 3, Eigen::ColMajor> upsampleX2_nearest(
		const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& field);

/**
 * Upsample the field such that the output is a field 2X larger than the original in each dimension using bilinear
 * filtering. This procedure uses a simple tent filter in each dimension to compute the values, i.e. bilinear filtering.
 * Conceptual example:
 * The influence coefficients for the voxels in the output field (o) fall off linearly from 1.0 at the
 * current input voxel (X) to 0.0 at it's neighbors (O).
 *    o     o    o     o    o     o
 *       O┈┈┈┈┈┈┈┈┈┈O┈┈┈┈┈┈┈┈┈┈O
 *    o  ┊  o    o     o    o  ┊  o
 *       ┊                     ┊
 *    o  ┊  o    o     o    o  ┊  o
 *       O          X          O
 *    o  ┊  o    o     o    o  ┊  o
 *       ┊                     ┊
 *    o  ┊  o    o     o    o  ┊  o
 *       O┈┈┈┈┈┈┈┈┈┈O┈┈┈┈┈┈┈┈┈┈O
 *    o     o    o     o    o     o
 *  Boundary voxels are processed as if the boundary values of the input repeat infinitely.
 *
 * @param field input field
 * @return upsampled field
 */
template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> upsampleX2_linear(
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& field);
/**
 * Upsample the field such that the output is a field 2X larger than the original in each dimension using bilinear
 * interpolation (LINEAR) strategy.
 * @overload
 */
//TODO
template<typename Scalar>
Eigen::Tensor<Scalar, 3, Eigen::ColMajor> upsampleX2_linear(
		const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& field);

/**
 * Downsample the provided matrix using a box filter such that each dimension of the downsampled field is half the
 * corresponding dimension of the input field.
 * @param field input field
 * @param downsampling_strategy strategy to use for downsampling
 * @return downsampled field
 */
template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> downsampleX2(
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& field,
		DownsamplingStrategy downsampling_strategy = DownsamplingStrategy::AVERAGE);
/**
 * Downsample the provided matrix using a box filter such that each dimension of the downsampled field is half the
 * corresponding dimension of the input field.
 * @overload
 */
template<typename Scalar>
Eigen::Tensor<Scalar, 3, Eigen::ColMajor> downsampleX2(
		const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& field,
		DownsamplingStrategy downsampling_strategy = DownsamplingStrategy::AVERAGE);

/**
 * Downsample the provided matrix using a box filter such that each dimension of the downsampled field is half the
 * corresponding dimension of the input field. Uses a simple box filter, i.e. each "downsampled" value will be the
 * average of it's source values in the input.
 *
 * Conceptual example (for 2d case):
 * ⎡1  2  4  5⎤
 * ⎢2  3  5  6⎥
 * ⎢1  3  6  7⎥
 * ⎣3  5  7  8⎦
 * yields
 * ⎡2  5⎤
 * ⎣3  7⎦
 *
 * @param field input field
 * @return downsampled field
 */
template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> downsampleX2_average(
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& field);
/**
 * Downsample the provided matrix using a box filter such that each dimension of the downsampled field is half the
 * corresponding dimension of the input field using simple averaging (AVERAGE strategy).
 * @overload
 */
//TODO
template<typename Scalar>
Eigen::Tensor<Scalar, 3, Eigen::ColMajor> downsampleX2_average(
		const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& field);

/**
 * Downsample the provided matrix using a box filter such that each dimension of the downsampled field is half the
 * corresponding dimension of the input field. Uses a tent filter, i.e. each "downsampled" value will be influenced by
 * values of the input weighted by the inverse ratio of their distance to the neighbor values.
 *
 * Conceptual example:
 * The influence coefficients for the source voxels in the input field (x) fall off linearly from 1.0 at the current
 * target voxel (X) to 0.0 at it's neighbors (O).
 *    o     o    o     o    o     o
 *       O┈┈┈┈┈┈┈┈┈┈O┈┈┈┈┈┈┈┈┈┈O
 *    o  ┊  o    o     o    o  ┊  o
 *       ┊                     ┊
 *    o  ┊  o    o     o    o  ┊  o
 *       O          X          O
 *    o  ┊  o    o     o    o  ┊  o
 *       ┊                     ┊
 *    o  ┊  o    o     o    o  ┊  o
 *       O┈┈┈┈┈┈┈┈┈┈O┈┈┈┈┈┈┈┈┈┈O
 *    o     o    o     o    o     o
 *  Boundary voxels are processed as if the boundary values of the input repeat infinitely.
 *
 * @param field input field
 * @return downsampled field
 */
template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> downsampleX2_linear(
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& field);
/**
 * Downsample the provided matrix using a box filter such that each dimension of the downsampled field is half the
 * corresponding dimension of the input field using bilinear interpolation (LINEAR strategy).
 * @overload
 */
//TODO
template<typename Scalar>
Eigen::Tensor<Scalar, 3, Eigen::ColMajor> downsampleX2_linear(
		const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& field);

} // namespace math
