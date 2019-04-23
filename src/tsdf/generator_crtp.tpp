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
#include "generator_crtp.hpp"
#include "../error_handling/throw_assert.hpp"
#include "interpolation_method.hpp"

namespace tsdf {

namespace eig = Eigen;

template<typename Generator, typename ScalarContainer>
GeneratorCRTP<Generator, ScalarContainer>::GeneratorCRTP(const Parameters<ScalarContainer>& parameters) :
		parameters(parameters) {
}

/**
 * @brief Generate a discrete implicit TSDF (Truncated Signed Distance Function) from the give depth image presumed to
 * have been taken at the specified camera pose.
 * @details Each voxel will contain the distance to the nearest surface, in voxels,
 * truncated to +/- 1.0. Uses voxel size, truncation bounds, and other parameters that the generator was initialized
 * with. For 2D versions of the algorithm, a coordinate is specified to limit usage of the depth image to a single pixel
 * row.
 * @param depth_image a 16-bit depth image
 * @param camera_pose pose of the camera relative to world
 * @param image_y_coordinate - (2D case only) row of the depth image to use
 * @return the generated TSDF voxel grid.
 */
template<typename Generator, typename ScalarContainer>
ScalarContainer GeneratorCRTP<Generator, ScalarContainer>::generate(
		const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic, eig::ColMajor>& depth_image,
		const eig::Matrix<ContainerScalar, 4, 4, eig::ColMajor>& camera_pose,
		int image_y_coordinate) const {

	switch (parameters.interpolation_method) {
	case InterpolationMethod::NONE:
		return static_cast<Generator const&>(*this).generate__none(depth_image, camera_pose, image_y_coordinate);
		//TODO: some time in the distant future, maybe, ...
//	case InterpolationMethod::BILINEAR_IMAGE_SPACE:
//		static_cast<Generator const&>(*this).generate__bilinear_image_space(depth_image,camera_pose,image_y_coordinate);
//		break;
//	case InterpolationMethod::BILINEAR_VOXEL_SPACE:
//		static_cast<Generator const&>(*this).generate__bilinear_tsdf_space(depth_image,camera_pose,image_y_coordinate);
//		break;
	case InterpolationMethod::EWA_IMAGE_SPACE:
		return static_cast<Generator const&>(*this).generate__ewa_image_space(depth_image, camera_pose,
				image_y_coordinate);
		break;
	case InterpolationMethod::EWA_VOXEL_SPACE:
		return static_cast<Generator const&>(*this).generate__ewa_voxel_space(depth_image, camera_pose,
				image_y_coordinate);
		break;
	case InterpolationMethod::EWA_VOXEL_SPACE_INCLUSIVE:
		return static_cast<Generator const&>(*this).generate__ewa_voxel_space_inclusive(depth_image, camera_pose,
				image_y_coordinate);
		break;
	default:
		throw_assert(false,
				"Unknown InterpolationMethod enum value, " << static_cast<int>(parameters.interpolation_method))
		;
	}
}

}  // namespace tsdf

