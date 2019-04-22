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

//local
#include "../math/extent.hpp"
#include "common.hpp"

namespace eig = Eigen;

namespace tsdf{

template<typename Scalar>
inline eig::Matrix<Scalar,3,3,eig::ColMajor>
compute_covariance_camera_space(Scalar voxel_size, const eig::Matrix<Scalar,4,4>& camera_pose,
		Scalar gaussian_covariance_scale = 1.0f) {
	eig::Matrix<Scalar,3,3,eig::ColMajor> camera_rotation_matrix = camera_pose.block(0, 0, 3, 3);
	eig::Matrix<Scalar,3,3,eig::ColMajor> covariance_voxel_sphere_world_space = eig::Matrix<Scalar,3,3>::Identity() *
			(voxel_size * gaussian_covariance_scale);
	eig::Matrix<Scalar,3,3,eig::ColMajor> covariance_camera_space =
			camera_rotation_matrix * covariance_voxel_sphere_world_space * camera_rotation_matrix.transpose();
	return covariance_camera_space;
}

template<typename Scalar>
inline bool compute_sampling_bounds(
		math::Extent2d& extent,
		const eig::Matrix<Scalar,2,1>& bounds_max,
		const eig::Matrix<Scalar,2,1>& voxel_image,
		const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& depth_image) {
	// compute sampling bounds
	int x_sample_start = static_cast<int>(voxel_image(0) - bounds_max(0));
	int x_sample_end = static_cast<int>(std::ceil(voxel_image(0) + bounds_max(0) + 1.0f));
	int y_sample_start = static_cast<int>(voxel_image(1) - bounds_max(1));
	int y_sample_end = static_cast<int>(std::ceil(voxel_image(1) + bounds_max(1) + 1.0f));

	// check that at least some samples within sampling range fall within the depth image
	if (x_sample_start >= depth_image.cols() || x_sample_end <= 0
			|| y_sample_start >= depth_image.rows() || y_sample_end <= 0) {
		return false;
	}

	// limit sampling bounds to image bounds
	extent.x_start = std::max(0, x_sample_start);
	extent.x_end = std::min(static_cast<int>(depth_image.cols()), x_sample_end);
	extent.y_start = std::max(0, y_sample_start);
	extent.y_end = std::min(static_cast<int>(depth_image.rows()), y_sample_end);
	return true;
}

template<typename Scalar>
inline bool compute_sampling_bounds_inclusive(
		math::Extent2d& extent,
		const eig::Matrix<Scalar,2,1>& bounds_max,
		const eig::Matrix<Scalar,2,1>& voxel_image,
		const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& depth_image) {
	// compute sampling bounds
	extent.x_start = static_cast<int>(voxel_image(0) - bounds_max(0));
	extent.x_end = static_cast<int>(std::ceil(voxel_image(0) + bounds_max(0) + 1.0f));
	extent.y_start = static_cast<int>(voxel_image(1) - bounds_max(1));
	extent.y_end = static_cast<int>(std::ceil(voxel_image(1) + bounds_max(1) + 1.0f));

	//TODO: potential speedup -- remove check here and make function void -- we're already checking for "out-of-bounds" voxels
	// check that at least some samples within sampling range fall within the depth image
	if (extent.x_start >= depth_image.cols() || extent.x_end <= 0
			|| extent.y_start >= depth_image.rows() || extent.y_end <= 0) {
		return false;
	}
	return true;
}




template<typename Scalar>
inline
Scalar compute_voxel_EWA_image_space(
		const math::Extent2d& sampling_bounds,
		const eig::Matrix<Scalar,2,1>& voxel_image,
		const eig::Matrix<Scalar,3,1>& voxel_camera,
		const eig::Matrix<Scalar,2,2>& ellipse_matrix,
		const Scalar& squared_radius_threshold,
		const Scalar& depth_unit_ratio,
		const Scalar& narrow_band_half_width,
		const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& depth_image){
	Scalar weights_sum = static_cast<Scalar>(0.0);
	Scalar depth_sum = static_cast<Scalar>(0.0);

	// collect sample readings
	for (int x_sample = sampling_bounds.x_start; x_sample < sampling_bounds.x_end; x_sample++) {
		for (int y_sample = sampling_bounds.y_start; y_sample < sampling_bounds.y_end; y_sample++) {
			eig::Matrix<Scalar,2,1> sample_centered;
			sample_centered <<
					static_cast<Scalar>(x_sample) - voxel_image(0),
					static_cast<Scalar>(y_sample) - voxel_image(1);
			Scalar dist_sq = sample_centered.transpose() * ellipse_matrix * sample_centered;
			//TODO: potential speedup -- remove check
			if (dist_sq > squared_radius_threshold) {
				continue;
			}
			Scalar weight = std::exp(static_cast<Scalar>(-0.5) * dist_sq);
			Scalar surface_depth = static_cast<Scalar>(depth_image(y_sample, x_sample)) * depth_unit_ratio;
			if (surface_depth <= static_cast<Scalar>(0.0)) {
				continue;
			}
			depth_sum += weight * surface_depth;
			weights_sum += weight;
		}
	}
	if (depth_sum <= static_cast<Scalar>(0.0)) {
		return static_cast<Scalar>(1.0);
	}

	Scalar final_depth = depth_sum / weights_sum;

	// signed distance from surface to voxel along camera axis
	// TODO: try with "along ray" and compare. Newcombe et al. in KinectFusion claim there won't be a difference...
	Scalar signed_distance = final_depth - voxel_camera[2];

	return compute_TSDF_value(signed_distance, narrow_band_half_width);
}

template<typename Scalar>
inline
Scalar compute_voxel_EWA_voxel_space(
		const math::Extent2d& sampling_bounds,
		const eig::Matrix<Scalar,2,1>& voxel_image,
		const eig::Matrix<Scalar,3,1>& voxel_camera,
		const eig::Matrix<Scalar,2,2>& ellipse_matrix,
		const Scalar& squared_radius_threshold,
		const Scalar& depth_unit_ratio,
		const Scalar& narrow_band_half_width,
		const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& depth_image){
	Scalar weights_sum = static_cast<Scalar>(0.0);
	Scalar TSDF_sum = static_cast<Scalar>(0.0);

	// collect sample readings
	for (int x_sample = sampling_bounds.x_start; x_sample < sampling_bounds.x_end; x_sample++) {
		for (int y_sample = sampling_bounds.y_start; y_sample < sampling_bounds.y_end; y_sample++) {
			eig::Matrix<Scalar,2,1> sample_centered;
			sample_centered <<
					static_cast<Scalar>(x_sample) - voxel_image(0),
					static_cast<Scalar>(y_sample) - voxel_image(1);
			Scalar dist_sq = sample_centered.transpose() * ellipse_matrix * sample_centered;
			if (dist_sq > squared_radius_threshold) {
				continue;
			}
			Scalar weight = std::exp(static_cast<Scalar>(-0.5) * dist_sq);
			Scalar surface_depth = static_cast<Scalar>(depth_image(y_sample, x_sample)) * depth_unit_ratio;
			if (surface_depth <= static_cast<Scalar>(0.0)) {
				continue;
			}
			Scalar signed_distance = surface_depth - voxel_camera[2];
			TSDF_sum += weight * compute_TSDF_value(signed_distance, narrow_band_half_width);
			weights_sum += weight;
		}
	}

	if (weights_sum == static_cast<Scalar>(0.0)) {
		return static_cast<Scalar>(1.0);
	}

	return TSDF_sum / weights_sum;
}


template<typename Scalar>
inline
Scalar compute_voxel_EWA_voxel_space_inclusive(
		const math::Extent2d& sampling_bounds,
		const eig::Matrix<Scalar,2,1>& voxel_image,
		const eig::Matrix<Scalar,3,1>& voxel_camera,
		const eig::Matrix<Scalar,2,2>& ellipse_matrix,
		const Scalar& squared_radius_threshold,
		const Scalar& depth_unit_ratio,
		const Scalar& narrow_band_half_width,
		const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& depth_image){
	Scalar weights_sum = static_cast<Scalar>(0.0);
	Scalar TSDF_sum = static_cast<Scalar>(0.0);

	for (int x_sample = sampling_bounds.x_start; x_sample < sampling_bounds.x_end; x_sample++) {
		for (int y_sample = sampling_bounds.y_start; y_sample < sampling_bounds.y_end; y_sample++) {
			eig::Matrix<Scalar,2,1> sample_centered;
			sample_centered <<
					static_cast<Scalar>(x_sample) - voxel_image(0),
					static_cast<Scalar>(y_sample) - voxel_image(1);
			Scalar dist_sq = sample_centered.transpose() * ellipse_matrix * sample_centered;
			//TODO: potential speedup -- remove range checking
			if (dist_sq > squared_radius_threshold) {
				continue;
			}
			Scalar weight = std::exp(static_cast<Scalar>(-0.5) * dist_sq);

			if (y_sample < 0 || y_sample >= depth_image.rows() ||
					x_sample < 0 || x_sample >= depth_image.cols()) {
				TSDF_sum += weight;// * 1.0;
			} else {
				Scalar surface_depth = static_cast<Scalar>(depth_image(y_sample, x_sample));
				if (surface_depth <= static_cast<Scalar>(0.0)) {
					continue;
				}
				Scalar signed_distance = surface_depth - voxel_camera[2];
				TSDF_sum += weight * compute_TSDF_value(signed_distance, narrow_band_half_width);
			}
			weights_sum += weight;
		}
	}
	//TODO: potential speedup -- is it even possible for this condition to be true?
	if (weights_sum == static_cast<Scalar>(0.0)) {
		return static_cast<Scalar>(1.0);
	}

	return TSDF_sum / weights_sum;
}


}//namespace tsdf


