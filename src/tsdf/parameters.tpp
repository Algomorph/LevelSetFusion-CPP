/*
 * parameters.tpp
 *
 *  Created on: Apr 23, 2019
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

#include "parameters.hpp"

namespace tsdf{

template<typename ScalarContainer>
Parameters<ScalarContainer>::Parameters(Scalar depth_unit_ratio,
		Mat3 projection_matrix,
		Scalar near_clipping_distance,
		Coordinates array_offset,
		Coordinates field_shape,
		Scalar voxel_size,
		int narrow_band_width_voxels,
		FilteringMethod interpolation_method,
		Scalar smoothing_factor
		) :
		depth_unit_ratio(depth_unit_ratio),
		projection_matrix(projection_matrix),
		near_clipping_distance(near_clipping_distance),
		array_offset(array_offset),
		field_shape(field_shape),
		voxel_size(voxel_size),
		narrow_band_width_voxels(narrow_band_width_voxels),
		interpolation_method(interpolation_method),
		smoothing_factor(smoothing_factor)
		{}

}  // namespace tsdf


