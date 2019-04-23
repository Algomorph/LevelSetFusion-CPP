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

//stdlib
#include <atomic>

//local
#include <lsf_config.h>
#include "generator.hpp"
#include "common.hpp"
#include "ewa_common.hpp"
#include "../math/conics.hpp"
#ifdef SDF_GENERATION_CONSOLE_PROGRESS_REPORTS
#include "../console/progress_bar.hpp"
#endif

namespace tsdf {

template<typename Scalar>
Eigen::Tensor<Scalar, 3, eig::ColMajor>
Generator<eig::Tensor<Scalar, 3, eig::ColMajor> >::generate__none(
		const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic, eig::ColMajor>& depth_image,
		const eig::Matrix<Scalar, 4, 4, eig::ColMajor>& camera_pose,
		int image_y_coordinate) const {
	const Parameters<Ts>& p = this->parameters;

	Ts field(p.field_shape.x, p.field_shape.y, p.field_shape.z);
	std::fill_n(field.data(), field.size(), (Scalar) 1.0);
	Scalar narrow_band_half_width = (static_cast<Scalar>(p.narrow_band_width_voxels) / 2.) * p.voxel_size;

	Scalar w_voxel = 1.0f;

	int voxel_count = static_cast<int>(field.size());

	int y_stride = p.field_shape.x;
	int z_stride = y_stride * p.field_shape.y;

#pragma omp parallel for
	for (int i_element = 0; i_element < voxel_count; i_element++) {
		int z_field = i_element / z_stride;
		int remainder = i_element % z_stride;
		int y_field = remainder / y_stride;
		int x_field = remainder % y_stride;

		Scalar x_voxel = (x_field + p.array_offset.x) * p.voxel_size;
		Scalar y_voxel = (y_field + p.array_offset.y) * p.voxel_size;
		Scalar z_voxel = (z_field + p.array_offset.z) * p.voxel_size;

		Vec4 voxel_world;
		voxel_world << x_voxel, y_voxel, z_voxel, w_voxel;
		Vec3 voxel_camera = (camera_pose * voxel_world).topRows(3);
		if (voxel_camera(2) <= p.near_clipping_distance) {
			continue;
		}

		Vec2 voxel_image = ((p.projection_matrix * voxel_camera) / voxel_camera[2]).topRows(2);

		if (is_voxel_out_of_bounds(voxel_image, depth_image, 0)) {
			continue;
		}

		// ray distance from camera to voxel center
		// TODO: there is difference between voxel_camera.norm() and voxel_camera(2).
		Scalar ray_distance = voxel_camera(2);

		int image_x_coordinate = int(voxel_image(0) + 0.5);
		int image_y_coordinate = int(voxel_image(1) + 0.5);
		Scalar depth = static_cast<float>(depth_image(image_y_coordinate, image_x_coordinate)) * p.depth_unit_ratio;

		if (depth <= 0.0f) {
			continue;
		}

		Scalar signed_distance_to_voxel_along_camera_ray = depth - ray_distance;

		field(x_field, y_field, z_field) = compute_TSDF_value(signed_distance_to_voxel_along_camera_ray, narrow_band_half_width);

	}

	return field;
}

template<typename Scalar>
template<typename SamplingBoundsFunction, typename VoxelValueFunction>
Eigen::Tensor<Scalar, 3, eig::ColMajor>
Generator<eig::Tensor<Scalar, 3, eig::ColMajor> >::generate__ewa_aux(SamplingBoundsFunction&& compute_sampling_bounds,
		VoxelValueFunction&& compute_voxel_value,
		const Eigen::Matrix<unsigned short, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& depth_image,
		const Eigen::Matrix<Scalar, 4, 4, Eigen::ColMajor>& camera_pose,
		int image_y_coordinate) const {
	const Parameters<Ts>& p = this->parameters;

	Ts field(p.field_shape.x, p.field_shape.y, p.field_shape.z);
	std::fill_n(field.data(), field.size(), (Scalar) 1.0);
	Scalar narrow_band_half_width = (static_cast<Scalar>(p.narrow_band_width_voxels) / 2.0) * p.voxel_size;

	Scalar w_voxel = (Scalar) 1.0;

	Mat3 covariance_camera_space =
			compute_covariance_camera_space(p.voxel_size, camera_pose, p.smoothing_factor);

	Mat2 image_space_scaling_matrix = p.projection_matrix.block(0, 0, 2, 2);

	Scalar squared_radius_threshold = (Scalar) 4.0 * p.voxel_size * p.smoothing_factor;
	int voxel_count = static_cast<int>(field.size());

	int y_stride = p.field_shape.x;
	int z_stride = y_stride * p.field_shape.y;

#ifdef SDF_GENERATION_CONSOLE_PROGRESS_REPORTS
	std::atomic<long> processed_voxel_count(0);
	long report_interval = voxel_count / 100; //1% intervals
	double last_reported_progress = 0.0f;
	console::ProgressBar progress_bar;
#endif

#pragma omp parallel for
	for (int i_element = 0; i_element < voxel_count; i_element++) {
		int z_field = i_element / z_stride;
		int remainder = i_element % z_stride;
		int y_field = remainder / y_stride;
		int x_field = remainder % y_stride;

		Scalar x_voxel = (x_field + p.array_offset.x) * p.voxel_size;
		Scalar y_voxel = (y_field + p.array_offset.y) * p.voxel_size;
		Scalar z_voxel = (z_field + p.array_offset.z) * p.voxel_size;

		Vec4 voxel_world;
		voxel_world << x_voxel, y_voxel, z_voxel, w_voxel;

		Vec3 voxel_camera = (camera_pose * voxel_world).topRows(3);
		if (voxel_camera(2) <= p.near_clipping_distance) {
			continue;
		}

		Vec2 voxel_image = ((p.projection_matrix * voxel_camera) / voxel_camera[2]).topRows(2);

		if (is_voxel_out_of_bounds(voxel_image, depth_image)) {
			continue;
		}

		// ray distance from camera to voxel center
		Scalar ray_distance = voxel_camera.norm();
		// squared distance along optical axis from camera to voxel
		Scalar z_cam_squared = voxel_camera(2) * voxel_camera(2);
		Scalar inv_z_cam = 1.0f / voxel_camera(2);

		// Jacobian for first-order tailor approximation of affine transform at pixel
		Mat3 projection_jacobian;
		projection_jacobian <<
				inv_z_cam, 0.0f, -voxel_camera(0) / z_cam_squared,
				0.0f, inv_z_cam, -voxel_camera(1) / z_cam_squared,
				voxel_camera(0) / ray_distance, voxel_camera(1) / ray_distance, voxel_camera(2) / ray_distance;

		Mat3 remapped_covariance =
				projection_jacobian * covariance_camera_space * projection_jacobian.transpose();

		// Resampling filter combines the covariance matrices of the
		// warped prefilter (remapped_covariance) and reconstruction filter (identity) of by adding them.
		Mat2 final_covariance =
				image_space_scaling_matrix * remapped_covariance.block(0, 0, 2, 2)
						* image_space_scaling_matrix.transpose()
						+ Mat2::Identity();
		Mat2 ellipse_matrix = final_covariance.inverse();

		Vec2 bounds_max = math::compute_centered_ellipse_bound_points(ellipse_matrix,
				squared_radius_threshold);

		// compute sampling bounds
		math::Extent2d sampling_bounds;
		if (!std::forward<SamplingBoundsFunction>(compute_sampling_bounds)(sampling_bounds, bounds_max, voxel_image,
				depth_image)) {
			continue;
		}

		field(x_field, y_field, z_field) = std::forward<VoxelValueFunction>(compute_voxel_value)(
				sampling_bounds, voxel_image, voxel_camera, ellipse_matrix,
				squared_radius_threshold, p.depth_unit_ratio, narrow_band_half_width,
				depth_image);

#ifdef SDF_GENERATION_CONSOLE_PROGRESS_REPORTS
		++processed_voxel_count;
#pragma omp critical
		if (processed_voxel_count % report_interval == 0) {
			double current_progress = static_cast<Scalar>(processed_voxel_count) / static_cast<Scalar>(voxel_count);
			double progress_increment = current_progress - last_reported_progress;
			last_reported_progress = current_progress;
			progress_bar.update(progress_increment);
			progress_bar.print();
		}
#endif
	}

#ifdef SDF_GENERATION_CONSOLE_PROGRESS_REPORTS
	std::cout << std::endl;
#endif
	return field;

}

template<typename Scalar>
Eigen::Tensor<Scalar, 3, eig::ColMajor>
Generator<eig::Tensor<Scalar, 3, eig::ColMajor>>::generate__ewa_image_space(
		const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic, eig::ColMajor>& depth_image,
		const eig::Matrix<Scalar, 4, 4, eig::ColMajor>& camera_pose,
		int image_y_coordinate) const {
	return this->generate__ewa_aux(compute_sampling_bounds<Scalar>, compute_voxel_EWA_image_space<Scalar>,
			depth_image, camera_pose, image_y_coordinate);
}

template<typename Scalar>
Eigen::Tensor<Scalar, 3, eig::ColMajor>
Generator<eig::Tensor<Scalar, 3, eig::ColMajor>>::generate__ewa_voxel_space(
		const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic, eig::ColMajor>& depth_image,
		const eig::Matrix<Scalar, 4, 4, eig::ColMajor>& camera_pose,
		int image_y_coordinate) const {
	return this->generate__ewa_aux(compute_sampling_bounds<Scalar>, compute_voxel_EWA_voxel_space<Scalar>,
			depth_image, camera_pose, image_y_coordinate);
}

template<typename Scalar>
Eigen::Tensor<Scalar, 3, eig::ColMajor>
Generator<eig::Tensor<Scalar, 3, eig::ColMajor>>::generate__ewa_voxel_space_inclusive(
		const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic, eig::ColMajor>& depth_image,
		const eig::Matrix<Scalar, 4, 4, eig::ColMajor>& camera_pose,
		int image_y_coordinate) const {
	return this->generate__ewa_aux(compute_sampling_bounds_inclusive<Scalar>,
			compute_voxel_EWA_voxel_space_inclusive<Scalar>,
			depth_image, camera_pose, image_y_coordinate);
}

}

