/*
 * generator.tpp
 *
 *  Created on: Apr 19, 2019
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

//local
#include "generator.hpp"
#include "../error_handling/throw_assert.hpp"

namespace tsdf {

namespace eig = Eigen;

template<typename ScalarContainer>
Parameters<ScalarContainer>::Parameters(Scalar depth_unit_ratio,
		Mat3 projection_matrix,
		Scalar near_clipping_distance,
		Coordinates array_offset,
		Coordinates field_shape,
		Scalar voxel_size,
		int narrow_band_width_voxels,
		InterpolationMethod interpolation_method,
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


template<typename Generator, typename ScalarContainer>
GeneratorCRTP<Generator, ScalarContainer>::GeneratorCRTP(const Parameters<ScalarContainer>& parameters) :
		parameters(parameters) {
}

template<typename Generator, typename ScalarContainer>
ScalarContainer GeneratorCRTP<Generator, ScalarContainer>::generate(
		const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic, eig::ColMajor>& depth_image,
		const eig::Matrix<ContainerScalar, 4, 4, eig::ColMajor>& camera_pose,
		int image_y_coordinate) const {

	switch (parameters.interpolation_method) {
	case InterpolationMethod::NONE:
		return static_cast<Generator const&>(*this).generate__none(depth_image,camera_pose,image_y_coordinate);
		//TODO: some time in the distant future, maybe, ...
//	case InterpolationMethod::BILINEAR_IMAGE_SPACE:
//		static_cast<Generator const&>(*this).generate__bilinear_image_space(depth_image,camera_pose,image_y_coordinate);
//		break;
//	case InterpolationMethod::BILINEAR_VOXEL_SPACE:
//		static_cast<Generator const&>(*this).generate__bilinear_tsdf_space(depth_image,camera_pose,image_y_coordinate);
//		break;
	case InterpolationMethod::EWA_IMAGE_SPACE:
		return static_cast<Generator const&>(*this).generate__ewa_image_space(depth_image,camera_pose,image_y_coordinate);
		break;
	case InterpolationMethod::EWA_VOXEL_SPACE:
		return static_cast<Generator const&>(*this).generate__ewa_voxel_space(depth_image,camera_pose,image_y_coordinate);
		break;
	case InterpolationMethod::EWA_VOXEL_SPACE_INCLUSIVE:
		return static_cast<Generator const&>(*this).generate__ewa_voxel_space_inclusive(depth_image,camera_pose,image_y_coordinate);
		break;
	default:
		throw_assert(false, "Unknown InterpolationMethod enum value, " << static_cast<int>(parameters.interpolation_method));
	}
}


}  // namespace tsdf

