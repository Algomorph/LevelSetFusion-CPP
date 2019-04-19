/*
 * generator_tensor.tpp
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

//local
#include "generator.tpp"
#include "common.hpp"

namespace tsdf{

template<typename Scalar>
Eigen::Tensor<Scalar, 3, eig::ColMajor>
Generator<eig::Tensor<Scalar, 3, eig::ColMajor> >::generate__none(
			const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic, eig::ColMajor>& depth_image,
			const eig::Matrix<Scalar, 4, 4, eig::ColMajor>& camera_pose,
			int image_y_coordinate){
	throw_assert(false, "Not implemented");
}

template<typename Scalar>
Eigen::Tensor<Scalar, 3, eig::ColMajor>
Generator<eig::Tensor<Scalar, 3, eig::ColMajor>>::generate__ewa_tsdf_space_inclusive(
			const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic, eig::ColMajor>& depth_image,
			const eig::Matrix<Scalar, 4, 4, eig::ColMajor>& camera_pose,
			int image_y_coordinate){
	throw_assert(false, "Not implemented");
}

}


