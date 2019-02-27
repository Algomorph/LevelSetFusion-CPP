//  ================================================================
//  Created by Gregory Kramida on 1/30/19.
//  Copyright (c) 2019 Gregory Kramida
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//  ================================================================

//standard library
#include <algorithm>
#include <atomic>
#include <cfloat>

//DEBUG
#include <limits>
#include <iostream>

//libraries
#include <unsupported/Eigen/CXX11/Tensor>

//local
#include <lsf_config.h>
#include "ewa.hpp"
#include "../math/conics.hpp"

#ifdef SDF_GENERATION_CONSOLE_PROGRESS_REPORTS
//local
#include "../console/progress_bar.hpp"
#endif

//TODO: clean out statements & commented code marked as //DEBUG after this is fully fixed

namespace tsdf {

static inline eig::Matrix3f compute_covariance_camera_space(float voxel_size, const eig::Matrix4f& camera_pose) {
	eig::Matrix3f camera_rotation_matrix = camera_pose.block(0, 0, 3, 3);
	eig::Matrix3f covariance_voxel_sphere_world_space = eig::Matrix3f::Identity() * voxel_size; // / 2;
	eig::Matrix3f covariance_camera_space =
			camera_rotation_matrix * covariance_voxel_sphere_world_space * camera_rotation_matrix.transpose();
	return covariance_camera_space;
}

static inline bool compute_sampling_bounds(
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

/**
 * Generate a 3D TSDF field from the provided depth image using Elliptical Weighed Average resampling approach.
 * A 3D Gaussian (standard deviation of 1 voxel) around every voxel is projected onto the depth image, the resulting
 * projection is convolved with a 2D Gaussian (standard deviation of 1 pixel), the resulting gaussian is used
 * as a weighted-average filter to sample from the depth image.
 * For details on EWA methods, refer to [1] and [2].
 * [1] P. S. Heckbert, “Fundamentals of Texture Mapping and Image Warping,” Jun. 1989.
 * [2] M. Zwicker, H. Pfister, J. Van Baar, and M. Gross, “EWA volume splatting,” in Visualization, 2001.
 *     VIS’01. Proceedings, 2001, pp. 29–538.
 * @param depth_image a 2D field of unsigned shorts, where every entry represents surface distance along the camera optical axis
 * @param depth_unit_ratio factor needed to convert depth values to meters, i.e. 0.001 for depth values with 1mm increments
 * @param camera_intrinsic_matrix intrinsic matrix of the camera, sometimes denoted as K (see Wikipedia for more info)
 * @param camera_pose camera extrinsic matrix (relative to world origin) / pose as a 4x4 matrix, which includes both
 * rotation matrix and translational components
 * @param array_offset offset of the minimum corner of the resulting SDF field from the world origin
 * @param field_shape field's shape, in voxels, dimensions in z,y,x order.
 * @param voxel_size size of every (2D) voxel's (edge) in meters
 * @param narrow_band_width_voxels width of the narrow band containing values in (-1.,1.0), or non-truncated values
 * @return resulting 2D square TSDF field
 */
eig::Tensor<float, 3> generate_3d_TSDF_field_from_depth_image_EWA(
		const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& depth_image,
		float depth_unit_ratio,
		const eig::Matrix3f& camera_intrinsic_matrix,
		const eig::Matrix4f& camera_pose,
		const eig::Vector3i& array_offset,
		const eig::Vector3i& field_shape,
		float voxel_size,
		int narrow_band_width_voxels) {

	eig::Tensor<float, 3> field(field_shape[0], field_shape[1], field_shape[2]);
	std::fill_n(field.data(), field.size(), 1.0f);
	float narrow_band_half_width = static_cast<float>(narrow_band_width_voxels / 2) * voxel_size;

	float w_voxel = 1.0f;

	eig::Matrix3f covariance_camera_space = compute_covariance_camera_space(voxel_size, camera_pose);

	eig::Matrix2f image_space_scaling_matrix = camera_intrinsic_matrix.block(0, 0, 2, 2);

	float squared_radius_threshold = 4.0f * voxel_size; //4.0f * (voxel_size / 2);
	int voxel_count = static_cast<int>(field.size());

	int y_stride = field.dimension(0);
	int z_stride = y_stride * field.dimension(1);

#ifdef SDF_GENERATION_CONSOLE_PROGRESS_REPORTS
	std::atomic<long> processed_voxel_count(0);
	long report_interval = voxel_count / 100; //1% intervals
	double last_reported_progress = 0.0f;
	console::ProgressBar progress_bar;
#endif

	//DEBUG
//	std::atomic<long> sampled_voxel_count(0);
//	std::atomic<long long> total_pixels_sampled(0.0);
//	std::atomic<int> max_voxel_samples(0);
//	std::atomic<int> min_voxel_samples(std::numeric_limits<int>::max());

#pragma omp parallel for
	for (int i_element = 0; i_element < voxel_count; i_element++) {
		// Any MatrixXf in Eigen is column-major
		// i_element = x * column_count + y
		div_t z_stride_division_result = std::div(i_element, z_stride);
		int z_field = z_stride_division_result.quot;
		div_t y_stride_division_result = std::div(z_stride_division_result.rem, y_stride);
		int y_field = y_stride_division_result.quot;
		int x_field = y_stride_division_result.rem;

		float x_voxel = (x_field + array_offset(0)) * voxel_size;
		float y_voxel = (y_field + array_offset(1)) * voxel_size;
		float z_voxel = (z_field + array_offset(2)) * voxel_size;

		eig::Vector4f voxel_world;
		voxel_world << x_voxel, y_voxel, z_voxel, w_voxel;

		eig::Vector3f voxel_camera = (camera_pose * voxel_world).topRows(3);
		if (voxel_camera(2) <= 0.0f) {
			continue;
		}

		// ray distance from camera to voxel center
		float ray_distance = voxel_camera.norm();
		// squared distance along optical axis from camera to voxel
		float z_cam_squared = voxel_camera(2) * voxel_camera(2);
		float inv_z_cam = 1.0f / voxel_camera(2);

		// Jacobian for first-order tailor approximation of affine transform at pixel
		eig::Matrix3f projection_jacobian;
		projection_jacobian <<
				inv_z_cam, 0.0f, -voxel_camera(0) / z_cam_squared,
				0.0f, inv_z_cam, -voxel_camera(1) / z_cam_squared,
				voxel_camera(0) / ray_distance, voxel_camera(1) / ray_distance, voxel_camera(2) / ray_distance;

		eig::Matrix3f remapped_covariance =
				projection_jacobian * covariance_camera_space * projection_jacobian.transpose();

		// Resampling filter combines the covariance matrices of the
		// warped prefilter (remapped_covariance) and reconstruction filter (identity) of by adding them.
		//TODO: This something around here is fishy
		eig::Matrix2f final_covariance =
				image_space_scaling_matrix * remapped_covariance.block(0, 0, 2, 2)
						* image_space_scaling_matrix.transpose()
						+ eig::Matrix2f::Identity();
		eig::Matrix2f ellipse_matrix = final_covariance.inverse();

		eig::Vector2f voxel_image = ((camera_intrinsic_matrix * voxel_camera) / voxel_camera[2]).topRows(2);

		eig::Vector2f bounds_max = math::compute_centered_ellipse_bound_points(ellipse_matrix,
				squared_radius_threshold);

		// compute sampling bounds
		int x_sample_start, x_sample_end, y_sample_start, y_sample_end;
		if (!compute_sampling_bounds(x_sample_start, x_sample_end, y_sample_start, y_sample_end,
				bounds_max, voxel_image, depth_image)) {
			continue;
		}

		float weights_sum = 0.0f;
		float depth_sum = 0.0f;
//
//		//DEBUG
//		int samples_counted = 0;

		// collect sample readings
		for (int x_sample = x_sample_start; x_sample < x_sample_end; x_sample++) {
			for (int y_sample = y_sample_start; y_sample < y_sample_end; y_sample++) {
				eig::Vector2f sample_centered;
				sample_centered <<
						static_cast<float>(x_sample) - voxel_image(0),
						static_cast<float>(y_sample) - voxel_image(1);
				float dist_sq = sample_centered.transpose() * ellipse_matrix * sample_centered;
				//potential speedup
				if (dist_sq > squared_radius_threshold) {
					continue;
				}
				float weight = std::exp(-0.5f * dist_sq);
				float surface_depth = static_cast<float>(depth_image(y_sample, x_sample)) * depth_unit_ratio;
				if (surface_depth <= 0.0f) {
					continue;
				}
				depth_sum += weight * surface_depth;
				weights_sum += weight;
				//samples_counted++;
			}
		}
		if (depth_sum <= 0.0) {
			continue;
		}

		float final_depth = depth_sum / weights_sum;

		// signed distance from surface to voxel along camera axis
		// TODO: try with "along ray" and compare. Newcombe et al. in KinectFusion claim there won't be a difference...
		float signed_distance = final_depth - voxel_camera[2];

		if (signed_distance < -narrow_band_half_width) {
			field(x_field, y_field, z_field) = -1.0;
		} else if (signed_distance > narrow_band_half_width) {
			field(x_field, y_field, z_field) = 1.0;
		} else {
			float tsdf_value = signed_distance / narrow_band_half_width;
			field(x_field, y_field, z_field) = tsdf_value;

			//DEBUG
//			if (std::abs(tsdf_value) < 0.1) {
//				sampled_voxel_count++;
//				total_pixels_sampled.fetch_add(samples_counted);
//				int prev_value = max_voxel_samples;
//				while (prev_value < samples_counted
//						&& !max_voxel_samples.compare_exchange_weak(prev_value, samples_counted))
//					;
//				prev_value = min_voxel_samples;
//				while (prev_value > samples_counted
//						&& !min_voxel_samples.compare_exchange_weak(prev_value, samples_counted))
//					;
//
//				if (samples_counted == 14) {
//					std::cout << "(EWA) " << i_element << ": " << x_field << ", " << y_field << ", " << z_field
//							<< " [" << x_voxel << ", " << y_voxel << ", " << z_voxel << "]" << "; ranges x, y: ["
//							<< x_sample_start << ", " << x_sample_end << ") , [" << y_sample_start << ", " << y_sample_end << ");"
//							<< " voxel_image: " << voxel_image << "; TSDF value: " << tsdf_value
//							<< "; bounds: " << bounds_max << std::endl;
//				}
//			}
		}

		//DEBUG: PROGRESS REPORTS
#ifdef SDF_GENERATION_CONSOLE_PROGRESS_REPORTS
		++processed_voxel_count;
#pragma omp critical
		if (processed_voxel_count % report_interval == 0) {
			double current_progress = static_cast<float>(processed_voxel_count) / static_cast<float>(voxel_count);
			double progress_increment = current_progress - last_reported_progress;
			last_reported_progress = current_progress;
			progress_bar.update(progress_increment);
			progress_bar.print();
		}
#endif
	}

	//DEBUG
//	std::cout << "[EWA STATISTICS]" << std::endl;
//	std::cout << "Average samples per voxel: " << static_cast<double>(total_pixels_sampled)
//			/ static_cast<double>(sampled_voxel_count) << std::endl;
//	std::cout << "Minimum samples per voxel: " << min_voxel_samples << std::endl;
//	std::cout << "Maximum samples per voxel: " << max_voxel_samples << std::endl;

#ifdef SDF_GENERATION_CONSOLE_PROGRESS_REPORTS
	std::cout << std::endl;
#endif

	return field;
}

/**
 * Generate a square 2D TSDF field from a single row of the provided depth image using Elliptical Weighed Average
 * resampling approach.
 * A 3D Gaussian (standard deviation of 1 voxel) around every voxel is projected onto the depth image, the resulting
 * projection is convolved with a 2D Gaussian (standard deviation of 1 pixel), the resulting gaussian is used
 * as a weighted-average filter to sample from the depth image.
 * For details on EWA methods, refer to [1] and [2].
 * [1] P. S. Heckbert, “Fundamentals of Texture Mapping and Image Warping,” Jun. 1989.
 * [2] M. Zwicker, H. Pfister, J. Van Baar, and M. Gross, “EWA volume splatting,” in Visualization, 2001.
 *     VIS’01. Proceedings, 2001, pp. 29–538.
 * @param image_y_coordinate since this is a 2D simulation, only this row of pixels in the image will be used
 * @param depth_image a 2D field of unsigned shorts, where every entry represents surface distance along the camera optical axis
 * @param depth_unit_ratio factor needed to convert depth values to meters, i.e. 0.001 for depth values with 1mm increments
 * @param camera_intrinsic_matrix intrinsic matrix of the camera, sometimes denoted as K (see Wikipedia for more info)
 * @param camera_pose camera extrinsic matrix (relative to world origin) / pose as a 4x4 matrix, which includes both
 * rotation matrix and translational components
 * @param array_offset offset of the minimum corner of the resulting SDF field from the world origin
 * @param field_size field's side length, in voxels
 * @param voxel_size size of every (2D) voxel's (edge) in meters
 * @param narrow_band_width_voxels width of the narrow band containing values in (-1.,1.0), or non-truncated values
 * @return resulting 2D square TSDF field
 */
eig::MatrixXf generate_2d_TSDF_field_from_depth_image_EWA(
		int image_y_coordinate,
		const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& depth_image,
		float depth_unit_ratio,
		const eig::Matrix3f& camera_intrinsic_matrix,
		const eig::Matrix4f& camera_pose,
		const eig::Vector3i& array_offset,
		int field_size,
		float voxel_size,
		int narrow_band_width_voxels) {
	eig::MatrixXf field(field_size, field_size);
	std::fill_n(field.data(), field.size(), 1.0f);
	float narrow_band_half_width = static_cast<float>(narrow_band_width_voxels / 2) * voxel_size;

	float w_voxel = 1.0f;
	float y_voxel = 0.0f;

	eig::Matrix3f covariance_camera_space = compute_covariance_camera_space(voxel_size, camera_pose);

	float squared_radius_threshold = 4.0f * voxel_size; //4.0f * (voxel_size / 2);
	int matrix_size = static_cast<int>(field.size());

	eig::Matrix2f image_space_scaling_matrix = camera_intrinsic_matrix.block(0, 0, 2, 2);

#pragma omp parallel for
	for (int i_element = 0; i_element < matrix_size; i_element++) {
		// Any MatrixXf in Eigen is column-major
		// i_element = x * column_count + y
		div_t division_result = std::div(i_element, field_size);
		int x_field = division_result.quot;
		int y_field = division_result.rem;

		float x_voxel = (x_field + array_offset(0)) * voxel_size;
		float z_voxel = (y_field + array_offset(2)) * voxel_size;

		eig::Vector4f voxel_world;
		voxel_world << x_voxel, y_voxel, z_voxel, w_voxel;
		eig::Vector3f voxel_camera = (camera_pose * voxel_world).topRows(3);
		if (voxel_camera(2) <= 0.0f) {
			continue;
		}
		// ray distance from camera to voxel center
		float ray_distance = voxel_camera.norm();
		// squared distance along optical axis from camera to voxel
		float z_cam_squared = voxel_camera(2) * voxel_camera(2);
		float inv_z_cam = 1.0f / voxel_camera(2);

		// Jacobian for first-order tailor approximation of affine transform at pixel
		eig::Matrix3f projection_jacobian;
		projection_jacobian <<
				inv_z_cam, 0.0f, -voxel_camera(0) / z_cam_squared,
				0.0f, inv_z_cam, -voxel_camera(1) / z_cam_squared,
				voxel_camera(0) / ray_distance, voxel_camera(1) / ray_distance, voxel_camera(2) / ray_distance;
		eig::Matrix3f remapped_covariance =
				projection_jacobian * covariance_camera_space * projection_jacobian.transpose();

		// Resampling filter combines the covariance matrices of the
		// warped prefilter (remapped_covariance) and reconstruction filter (identity) of by adding them.
		eig::Matrix2f final_covariance = image_space_scaling_matrix * remapped_covariance.block(0, 0, 2, 2)
						* image_space_scaling_matrix.transpose() + eig::Matrix2f::Identity();
		eig::Matrix2f ellipse_matrix = final_covariance.inverse();

		eig::Vector2f voxel_image = ((camera_intrinsic_matrix * voxel_camera) / voxel_camera[2]).topRows(2);
		voxel_image(1) = image_y_coordinate;

		eig::Vector2f bounds_max = math::compute_centered_ellipse_bound_points(ellipse_matrix,
				squared_radius_threshold);

		// compute sampling bounds
		int x_sample_start, x_sample_end, y_sample_start, y_sample_end;
		if (!compute_sampling_bounds(x_sample_start, x_sample_end, y_sample_start, y_sample_end,
				bounds_max, voxel_image, depth_image)) {
			continue;
		}
		float weights_sum = 0.0f;
		float depth_sum = 0.0f;

		int samples_counted = 0;

		// collect sample readings
		for (int x_sample = x_sample_start; x_sample < x_sample_end; x_sample++) {
			for (int y_sample = y_sample_start; y_sample < y_sample_end; y_sample++) {
				eig::Vector2f sample_centered;
				sample_centered <<
						static_cast<float>(x_sample) - voxel_image(0),
						static_cast<float>(y_sample) - voxel_image(1);
				float dist_sq = sample_centered.transpose() * ellipse_matrix * sample_centered;
				if (dist_sq > squared_radius_threshold) {
					continue;
				}
				float weight = std::exp(-0.5f * dist_sq);
				float surface_depth = static_cast<float>(depth_image(y_sample, x_sample)) * depth_unit_ratio;
				if (surface_depth <= 0.0f) {
					continue;
				}
				depth_sum += weight * surface_depth;
				weights_sum += weight;
				samples_counted++;
			}
		}
		if (depth_sum <= 0.0) {
			continue;
		}
		float final_depth = depth_sum / weights_sum;

		// signed distance from surface to voxel along camera axis
		// TODO: try with "along ray" and compare. Newcombe et al. in KinectFusion claim there won't be a difference...
		float signed_distance = final_depth - voxel_camera[2];

		if (signed_distance < -narrow_band_half_width) {
			field(y_field, x_field) = -1.0;
		} else if (signed_distance > narrow_band_half_width) {
			field(y_field, x_field) = 1.0;
		} else {
			field(y_field, x_field) = signed_distance / narrow_band_half_width;
		}

	}

	return field;
}

/**
 * ======VISUALIZATION-ONLY VERSION=======================
 * Draw a visualization of voxel sampling over image space using Elliptical Weighed Average resampling approach.
 * See generate_3d_TSDF_field_from_depth_image_EWA() for method description. Draws the projected ellipses corresponding
 * to 1 standard deviation away from each voxel sample.
 *
 * @param depth_image a 2D field of unsigned shorts, where every entry represents surface distance along the camera optical axis
 * @param field a 3D field previously generated from the provided image
 * @param depth_unit_ratio factor needed to convert depth values to meters, i.e. 0.001 for depth values with 1mm increments
 * @param camera_intrinsic_matrix intrinsic matrix of the camera, sometimes denoted as K (see Wikipedia for more info)
 * @param camera_pose camera extrinsic matrix (relative to world origin) / pose as a 4x4 matrix, which includes both
 * rotation matrix and translational components
 * @param array_offset offset of the minimum corner of the resulting SDF field from the world origin
 * @param field_shape field's shape, in voxels, dimensions in z,y,x order.
 * @param voxel_size size of every (2D) voxel's (edge) in meters
 * @return resulting 2D square TSDF field
 */
eig::MatrixXuc generate_3d_TSDF_field_from_depth_image_EWA_viz(
		const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& depth_image,
		float depth_unit_ratio,
		const eig::Tensor<float, 3>& field,
		const eig::Matrix3f& camera_intrinsic_matrix,
		const eig::Matrix4f& camera_pose,
		const eig::Vector3i& array_offset,
		float voxel_size,
		int scale,
		float tsdf_threshold) {

	eig::MatrixXuc output_image = eig::MatrixXuc::Constant(depth_image.rows() * scale, depth_image.cols() * scale, 255);

	float w_voxel = 1.0f;

	eig::Matrix3f covariance_camera_space = compute_covariance_camera_space(voxel_size, camera_pose);

	float squared_radius_threshold = 2.0f * voxel_size; //4.0f * (voxel_size / 2);
	int field_size = static_cast<int>(field.size());

	int y_stride = field.dimension(0);
	int z_stride = y_stride * field.dimension(1);

	//DEBUG
	float closest_to_zero = std::numeric_limits<float>::max();
	long total_truncated = 0;
	long total_nontruncated = 0;

	eig::Matrix2f image_space_scaling_matrix = camera_intrinsic_matrix.block(0, 0, 2, 2);
	//DEBUG
	//int i_debug_element = 27918551;
	//int i_debug_element = 2666528;
	//int i_debug_element = -1;

	//DEBUG
//	float smallest_ellipse_size = std::numeric_limits<float>::max();
//	float largest_ellipse_size = 0.0f;

	for (int i_element = 0; i_element < field_size; i_element++) {

		div_t z_stride_division_result = std::div(i_element, z_stride);
		int z_field = z_stride_division_result.quot;
		div_t y_stride_division_result = std::div(z_stride_division_result.rem, y_stride);
		int y_field = y_stride_division_result.quot;
		int x_field = y_stride_division_result.rem;

		float tsdf_value = field(x_field, y_field, z_field);

		//DEBUG
		float abs_tsdf_value = std::abs(tsdf_value);
		if (std::abs(abs_tsdf_value - 1.0f) < FLT_EPSILON) {
			total_truncated++;
		} else {
			total_nontruncated++;
		}

		if (abs_tsdf_value > tsdf_threshold)
			continue;

		//DEBUG
		if (abs_tsdf_value < closest_to_zero) {
			closest_to_zero = std::abs(tsdf_value);
		}

		float x_voxel = (x_field + array_offset(0)) * voxel_size;
		float y_voxel = (y_field + array_offset(1)) * voxel_size;
		float z_voxel = (z_field + array_offset(2)) * voxel_size;

		eig::Vector4f voxel_world;
		voxel_world << x_voxel, y_voxel, z_voxel, w_voxel;

		eig::Vector3f voxel_camera = (camera_pose * voxel_world).topRows(3);

		// ray distance from camera to voxel center
		float ray_distance = voxel_camera.norm();
		// squared distance along optical axis from camera to voxel
		float z_cam_squared = voxel_camera(2) * voxel_camera(2);
		float inv_z_cam = 1.0f / voxel_camera(2);

		// Jacobian for first-order tailor approximation of affine transform at pixel
		eig::Matrix3f projection_jacobian;
		projection_jacobian <<
				inv_z_cam, 0.0f, -voxel_camera(0) / z_cam_squared,
				0.0f, inv_z_cam, -voxel_camera(1) / z_cam_squared,
				voxel_camera(0) / ray_distance, voxel_camera(1) / ray_distance, voxel_camera(2) / ray_distance;

		eig::Matrix3f remapped_covariance =
				projection_jacobian * covariance_camera_space * projection_jacobian.transpose();

		// Resampling filter combines the covariance matrices of the
		// warped prefilter (remapped_covariance) and reconstruction filter (identity) of by adding them.
		//TODO: why is the conic matrix not the inverse of the combined covariances?
		eig::Matrix2f final_covariance = image_space_scaling_matrix * remapped_covariance.block(0, 0, 2, 2) *
				image_space_scaling_matrix.transpose() + eig::Matrix2f::Identity();
		eig::Matrix2f ellipse_matrix = final_covariance.inverse();

//		float ellipse_size = ellipse_matrix(0, 0) * ellipse_matrix(1, 1)
//				- (ellipse_matrix(0, 1) * (ellipse_matrix(0, 1)));
//
//		//DEBUG
//		if (ellipse_size > largest_ellipse_size) {
//			largest_ellipse_size = ellipse_size;
//		}
//		if (ellipse_size < smallest_ellipse_size) {
//			smallest_ellipse_size = ellipse_size;
//		}

		eig::Vector2f voxel_image = ((camera_intrinsic_matrix * voxel_camera) / voxel_camera[2]).topRows(2);

		//DEBUG
//		if(i_element == i_debug_element){
//				float x_voxel = (x_field + array_offset(0)) * voxel_size;
//				float y_voxel = (y_field + array_offset(1)) * voxel_size;
//				float z_voxel = (z_field + array_offset(2)) * voxel_size;
//				std::cout << "(VIZ) " << i_element << ": " << x_field << ", " << y_field << ", " << z_field
//						<< " [" << x_voxel << ", " << y_voxel << ", " << z_voxel << "]"
//						<< "; TSDF value: " << tsdf_value << "; voxel_image: " << voxel_image
//						<< std::endl;
//				std::cout.flush();
//			math::draw_ellipse(output_image, voxel_image * scale, ellipse_matrix,
//								squared_radius_threshold * scale*scale /* ellipse_size*/, true);
//		}else{
		math::draw_ellipse(output_image, voxel_image * scale, ellipse_matrix,
				squared_radius_threshold * scale * scale /* ellipse_size*/);
		//DEBUG
//		}

	}

	//DEBUG
//	std::cout << "Closest TSDF value to zero: " << closest_to_zero /*<< ", smallest ellipse F: "
//	 << smallest_ellipse_size << ", largest ellipse F: " << largest_ellipse_size*/
//	<< "; Truncated SDF values: " << static_cast<double>(total_truncated) /
//			static_cast<double>(total_truncated + total_nontruncated) * 100.0 << "%"
//			<< std::endl;

	return output_image;
}

} // namespace tsdf
