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
#include <Eigen/Eigen>

namespace math{
/**
 * Upsample the field such that the output is a field 2X larger than the original in each dimension.
 * This procedure uses a simple box filter / no interpolation, i.e. simply copies the value to it's immediate "children"
 * in the upsampled version.
 * Conceptual example:
 * ⸢1  2⸣
 * ⸤3  4⸥
 * yields
 * ⸢ 1  1  2  2 ⸣
 * │1  1  2  2 │
 * │3  3  4  4 │
 * ⸤ 3  3  4  4 ⸥
 * @param field input field
 * @return upsampled field
 */
template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> upsampleX2(
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& field);

/**
 * @see upsampleX2 (previous definition)
 * @param field
 * @param field input field
 * @return upsampled field
 */
template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> upsampleX2(
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& field);

/**
 * Upsample the field such that the output is a field 2X larger than the original in each dimension using bilinear
 * filtering. This procedure uses a simple tent filter in each dimension to compute the values, i.e. bilinear filtering.
 * Conceptual example:
 * The influence coefficients for the voxels in the output field (small circles) fall off linearly from 1.0 at the
 * current input voxel (●) to 0.0 at it's neighbors (○).
 *    ⚬          ⚬       ⚬       ⚬        ⚬         ⚬
 *       ○┈┈┈┈┈┈┈┈┈○┈┈┈┈┈┈┈┈┈○
 *    ⚬    ┊  ⚬       ⚬       ⚬        ⚬   ┊  ⚬
 *       ┊                    ┊
 *    ⚬    ┊  ⚬       ⚬       ⚬        ⚬   ┊  ⚬
 *       ○                  ●                  ○
 *    ⚬    ┊  ⚬       ⚬       ⚬        ⚬   ┊  ⚬
 *       ┊                    ┊
 *    ⚬    ┊  ⚬       ⚬       ⚬        ⚬   ┊  ⚬
 *       ○┈┈┈┈┈┈┈┈┈○┈┈┈┈┈┈┈┈┈○
 *    ⚬          ⚬       ⚬       ⚬        ⚬         ⚬
 *  Boundary voxels are processed as if the boundary values of the input repeat infinitely.
 *
 * @param field input field
 * @return upsampled field
 */
template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> upsampleX2_bilinear(
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& field);


}// namespace math
