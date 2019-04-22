/*
 * common.hpp
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

namespace tsdf{

constexpr float near_clipping_distance = 0.05; //m

template<typename Scalar>
inline Scalar compute_TSDF_value(Scalar signed_distance, Scalar narrow_band_half_width){
	if (signed_distance < -narrow_band_half_width) {
		return (Scalar)-1.0;
	} else if (signed_distance > narrow_band_half_width) {
		return (Scalar)1.0;
	} else {
		return signed_distance / narrow_band_half_width;
	}
}

template<typename Scalar>
inline bool is_voxel_out_of_bounds(const Eigen::Matrix<Scalar,2,1>& voxel_image,
		const Eigen::Matrix<unsigned short, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& depth_image,
		int margin = 3){
	if (voxel_image(0) < -margin || voxel_image(0) >= depth_image.cols() + margin ||
			voxel_image(1) < -margin || voxel_image(1) >= depth_image.rows() + margin){
		return true;
	}
	return false;
}

}  // namespace tsdf



