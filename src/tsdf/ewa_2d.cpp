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

//libraries
#include <unsupported/Eigen/CXX11/Tensor>

//local
#include <lsf_config.h>
#include "ewa.hpp"
#include "ewa_common.hpp"
#include "../math/conics.hpp"

namespace tsdf {

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
eig::MatrixXf generate_TSDF_2D_EWA_image(
		int image_y_coordinate,
		const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& depth_image,
		float depth_unit_ratio,
		const eig::Matrix3f& camera_intrinsic_matrix,
		const eig::Matrix4f& camera_pose,
		const eig::Vector3i& array_offset,
		int field_size,
		float voxel_size,
		int narrow_band_width_voxels,
		float gaussian_covariance_scale) {
	eig::MatrixXf field(field_size, field_size);
	std::fill_n(field.data(), field.size(), 1.0f);
	float narrow_band_half_width = static_cast<float>(narrow_band_width_voxels / 2) * voxel_size;

	float w_voxel = 1.0f;
	float y_voxel = 0.0f;

	eig::Matrix3f covariance_camera_space =
			compute_covariance_camera_space(voxel_size, camera_pose, gaussian_covariance_scale);

	float squared_radius_threshold = 4.0f * voxel_size * gaussian_covariance_scale;
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

} // namespace tsdf
