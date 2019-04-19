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

template<typename Generator, typename ScalarContainer>
GeneratorCRTP<Generator, ScalarContainer>::GeneratorCRTP(const Parameters<ScalarContainer>& parameters) :
		parameters(parameters) {
}

template<typename Generator, typename ScalarContainer>
ScalarContainer GeneratorCRTP<Generator, ScalarContainer>::generate(
		const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic, eig::ColMajor>& depth_image,
		const eig::Matrix<ContainerScalar, 4, 4, eig::ColMajor>& camera_pose,
		int image_y_coordinate) {

	switch (parameters.interpolation_method) {
	case InterpolationMethod::NONE:
		static_cast<Generator const&>(*this).generate__none(depth_image,camera_pose,image_y_coordinate);
		break;
		//TODO: some time in the distant future, maybe, ...
//	case InterpolationMethod::BILINEAR_IMAGE_SPACE:
//		static_cast<Generator const&>(*this).generate__bilinear_image_space(depth_image,camera_pose,image_y_coordinate);
//		break;
//	case InterpolationMethod::BILINEAR_TSDF_SPACE:
//		static_cast<Generator const&>(*this).generate__bilinear_tsdf_space(depth_image,camera_pose,image_y_coordinate);
//		break;
//	case InterpolationMethod::EWA_IMAGE_SPACE:
//		static_cast<Generator const&>(*this).generate__ewa_image_space(depth_image,camera_pose,image_y_coordinate);
//		break;
//	case InterpolationMethod::EWA_TSDF_SPACE:
//		static_cast<Generator const&>(*this).generate__ewa_tsdf_space(depth_image,camera_pose,image_y_coordinate);
//		break;
	case InterpolationMethod::EWA_TSDF_SPACE_INCLUSIVE:
		static_cast<Generator const&>(*this).generate__ewa_tsdf_space_inclusive(depth_image,camera_pose,image_y_coordinate);
		break;
	default:
		throw_assert(false, "Unknown InterpolationMethod enum value, " << parameters.interpolation_method)
		;
	}
}


}  // namespace tsdf

