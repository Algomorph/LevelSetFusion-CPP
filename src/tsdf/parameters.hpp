/*
 * parameters.hpp
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

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "../math/typedefs.hpp"
#include "../math/container_traits.hpp"
#include "interpolation_method.hpp"

namespace tsdf{
template<typename ScalarContainer>
struct Parameters {
	typedef typename math::ContainerWrapper<ScalarContainer>::Coordinates Coordinates;
	typedef typename ScalarContainer::Scalar Scalar;
	typedef Eigen::Matrix<Scalar, 3, 3, Eigen::ColMajor> Mat3;
	Parameters(Scalar depth_unit_ratio = (Scalar)0.001,
			Mat3 projection_matrix = Mat3::Identity(),
			Scalar near_clipping_distance = (Scalar)0.05,
			Coordinates array_offset = Coordinates(-64),
			Coordinates field_shape = Coordinates(128),
			Scalar voxel_size = (Scalar)0.004,
			int narrow_band_width_voxels = 20,
			InterpolationMethod interpolation_method = InterpolationMethod::NONE,
			Scalar smoothing_factor = (Scalar)1.0
			);
	Scalar depth_unit_ratio = (Scalar)0.001; //meters
	Mat3 projection_matrix;
	Scalar near_clipping_distance = 0.05; //meters
	Coordinates array_offset = Coordinates(-64); //voxels
	Coordinates field_shape = Coordinates(128); //voxels
	Scalar voxel_size = 0.004; //meters
	int narrow_band_width_voxels = 20; //voxels
	InterpolationMethod interpolation_method = InterpolationMethod::NONE;
	Scalar smoothing_factor = (Scalar)1.0; // gaussian covariance scale for EWA
};

}  // namespace tsdf


