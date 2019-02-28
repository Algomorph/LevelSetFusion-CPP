/*
 * ewa_3d_viz.cpp
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
 * ======VISUALIZATION-ONLY VERSION=======================
 * Draw a visualization of voxel sampling over image space using Elliptical Weighed Average resampling approach.
 * See generate_3d_TSDF_field_from_depth_image_EWA() for method description. Draws the projected ellipses corresponding
 * to 1 standard deviation away from each voxel sample.
 *
 * CAUTION: this function is not parallelized, so it's very slow. Intended for visual debugging purposes.
 *
 * @param depth_image a 2D field of unsigned shorts, where every entry represents surface distance along the camera optical axis
 * @param field a 3D field previously generated from the provided image
 * @param depth_unit_ratio factor needed to convert depth values to meters, i.e. 0.001 for depth values with 1mm increments
 * @param camera_intrinsic_matrix intrinsic matrix of the camera, sometimes denoted as K (see Wikipedia for more info)
 * @param camera_pose camera extrinsic matrix (relative to world origin) / pose as a 4x4 matrix, which includes both
 * rotation matrix and translational components
 * @param array_offset offset of the minimum corner of the resulting SDF field from the world origin
 * @param field_shape field's shape, in voxels, dimensions in z,y,x order.
 * @param voxel_size size of every (3D) voxel's (edge) in meters
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

	eig::Matrix2f image_space_scaling_matrix = camera_intrinsic_matrix.block(0, 0, 2, 2);

	for (int i_element = 0; i_element < field_size; i_element++) {

		div_t z_stride_division_result = std::div(i_element, z_stride);
		int z_field = z_stride_division_result.quot;
		div_t y_stride_division_result = std::div(z_stride_division_result.rem, y_stride);
		int y_field = y_stride_division_result.quot;
		int x_field = y_stride_division_result.rem;

		float tsdf_value = field(x_field, y_field, z_field);

		if (std::abs(tsdf_value) > tsdf_threshold)
			continue;

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

		// Resampling filter combines the covariance matrices of the warped prefilter (scaled from camera to image space)
		// and reconstruction filter (identity) of by adding them.
		eig::Matrix2f final_covariance = image_space_scaling_matrix * remapped_covariance.block(0, 0, 2, 2) *
				image_space_scaling_matrix.transpose() + eig::Matrix2f::Identity();
		eig::Matrix2f ellipse_matrix = final_covariance.inverse();

		eig::Vector2f voxel_image = ((camera_intrinsic_matrix * voxel_camera) / voxel_camera[2]).topRows(2);
		math::draw_ellipse(output_image, voxel_image * scale, ellipse_matrix,
				squared_radius_threshold * scale * scale /* ellipse_size*/);

	}

	return output_image;
}

}//namespace tsdf





