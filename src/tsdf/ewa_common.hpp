/*
 * ewa_common.hpp
 *
 *  Created on: Feb 28, 2019
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

namespace eig = Eigen;

namespace tsdf{

inline eig::Matrix3f compute_covariance_camera_space(float voxel_size, const eig::Matrix4f& camera_pose, float gaussian_covariance_scale = 1.0f) {
	eig::Matrix3f camera_rotation_matrix = camera_pose.block(0, 0, 3, 3);
	eig::Matrix3f covariance_voxel_sphere_world_space = eig::Matrix3f::Identity() * (voxel_size * gaussian_covariance_scale);
	eig::Matrix3f covariance_camera_space =
			camera_rotation_matrix * covariance_voxel_sphere_world_space * camera_rotation_matrix.transpose();
	return covariance_camera_space;
}

inline float compute_TSDF_value(float signed_distance, float narrow_band_half_width){
	if (signed_distance < -narrow_band_half_width) {
		return -1.0;
	} else if (signed_distance > narrow_band_half_width) {
		return 1.0;
	} else {
		return signed_distance / narrow_band_half_width;
	}
}

inline bool compute_sampling_bounds(
		int& x_sample_start,
		int& x_sample_end,
		int& y_sample_start,
		int& y_sample_end,
		const eig::Vector2f& bounds_max,
		const eig::Vector2f& voxel_image,
		const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& depth_image) {
	// compute sampling bounds
	x_sample_start = static_cast<int>(voxel_image(0) - bounds_max(0));
	x_sample_end = static_cast<int>(std::ceil(voxel_image(0) + bounds_max(0) + 1.0f));
	y_sample_start = static_cast<int>(voxel_image(1) - bounds_max(1));
	y_sample_end = static_cast<int>(std::ceil(voxel_image(1) + bounds_max(1) + 1.0f));

	// check that at least some samples within sampling range fall within the depth image
	if (x_sample_start > depth_image.cols() || x_sample_end <= 0
			|| y_sample_start > depth_image.rows() || y_sample_end <= 0) {
		return false;
	}

	// limit sampling bounds to image bounds
	x_sample_start = std::max(0, x_sample_start);
	x_sample_end = std::min(static_cast<int>(depth_image.cols()), x_sample_end);
	y_sample_start = std::max(0, y_sample_start);
	y_sample_end = std::min(static_cast<int>(depth_image.rows()), y_sample_end);
	return true;
}

inline bool compute_sampling_bounds_inclusive(
		int& x_sample_start,
		int& x_sample_end,
		int& y_sample_start,
		int& y_sample_end,
		const eig::Vector2f& bounds_max,
		const eig::Vector2f& voxel_image,
		const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& depth_image) {
	// compute sampling bounds
	x_sample_start = static_cast<int>(voxel_image(0) - bounds_max(0));
	x_sample_end = static_cast<int>(std::ceil(voxel_image(0) + bounds_max(0) + 1.0f));
	y_sample_start = static_cast<int>(voxel_image(1) - bounds_max(1));
	y_sample_end = static_cast<int>(std::ceil(voxel_image(1) + bounds_max(1) + 1.0f));

	// check that at least some samples within sampling range fall within the depth image
	if (x_sample_start > depth_image.cols() || x_sample_end <= 0
			|| y_sample_start > depth_image.rows() || y_sample_end <= 0) {
		return false;
	}
	return true;
}

}//namespace tsdf


