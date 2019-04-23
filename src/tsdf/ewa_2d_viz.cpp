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
#include "ewa_viz.hpp"
#include "ewa_common.hpp"
#include "common.hpp"
#include "../math/conics.hpp"

namespace tsdf {


eig::MatrixXf sampling_area_heatmap_2D_EWA_image(
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
	eig::MatrixXf sampling_area_heatmap(field_size, field_size);
	std::fill_n(field.data(), field.size(), 1.0f);

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
		int x_field = i_element / field_size;
		int y_field = i_element % field_size;

		float x_voxel = (x_field + array_offset(0)) * voxel_size;
		float z_voxel = (y_field + array_offset(2)) * voxel_size;

		eig::Vector4f voxel_world;
		voxel_world << x_voxel, y_voxel, z_voxel, w_voxel;
		eig::Vector3f voxel_camera = (camera_pose * voxel_world).topRows(3);
		if (voxel_camera(2) <= near_clipping_distance) {
			continue;
		}

		eig::Vector2f voxel_image = ((camera_intrinsic_matrix * voxel_camera) / voxel_camera[2]).topRows(2);
		voxel_image(1) = image_y_coordinate;

		if(is_voxel_out_of_bounds(voxel_image,depth_image)){
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

		eig::Matrix2f final_covariance = (image_space_scaling_matrix * remapped_covariance.block(0, 0, 2, 2)
						* image_space_scaling_matrix.transpose()) + eig::Matrix2f::Identity();

		eig::Matrix2f ellipse_matrix = final_covariance.inverse();

		eig::Vector2f bounds_max = math::compute_centered_ellipse_bound_points(ellipse_matrix,
				squared_radius_threshold);

		// compute sampling bounds
		math::Extent2d sampling_bounds;
		if (!compute_sampling_bounds(sampling_bounds,bounds_max, voxel_image, depth_image)) {
			continue;
		}

		sampling_area_heatmap(y_field, x_field) = static_cast<float>(sampling_bounds.area());

	}

	return sampling_area_heatmap;
}

} // namespace tsdf


