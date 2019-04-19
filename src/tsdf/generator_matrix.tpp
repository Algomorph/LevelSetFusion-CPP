/*
 * generator_matrix.tpp
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
Eigen::Matrix<Scalar, eig::Dynamic, eig::Dynamic, eig::ColMajor>
Generator<eig::Matrix<Scalar, eig::Dynamic, eig::Dynamic, eig::ColMajor> >::generate__none(
			const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic, eig::ColMajor>& depth_image,
			const eig::Matrix<Scalar, 4, 4, eig::ColMajor>& camera_pose,
			int image_y_coordinate){

	const Parameters<Mat>& p = this->parameters;

	eig::MatrixXf field(p.field_shape.y, p.field_shape.x);
	float narrow_band_half_width = static_cast<float>(p.narrow_band_width_voxels / 2) * p.voxel_size;

	float w_voxel = 1.0f;
	float y_voxel = 0.0f;

	int matrix_size = static_cast<int>(field.size());

	int x_size = p.field_shape.x;
	int y_size = p.field_shape.y;

	#pragma omp parallel for
	for (int i_element = 0; i_element < matrix_size; i_element++) {
		// Any MatrixXf in Eigen is column-major
		// i_element = x * column_count + y
		int x_field = i_element / x_size;
		int y_field = i_element % y_size;

		float x_voxel = (x_field + p.array_offset.x) * p.voxel_size;
		float z_voxel = (y_field + p.array_offset.y) * p.voxel_size;

		eig::Vector4f voxel_world;
		voxel_world << x_voxel, y_voxel, z_voxel, w_voxel;
		eig::Vector3f voxel_camera = (camera_pose * voxel_world).topRows(3);
		if (voxel_camera(2) <= p.near_clipping_distance) {
			continue;
		}

		eig::Vector2f voxel_image = ((p.projection_matrix * voxel_camera) / voxel_camera[2]).topRows(2);
		voxel_image(1) = image_y_coordinate;

		if (is_voxel_out_of_bounds(voxel_image, depth_image, 0)) {
			continue;
		}

		// ray distance from camera to voxel center
		// TODO: there is difference between voxel_camera.norm() and voxel_camera[2].
		float ray_distance = voxel_camera(2);

		int image_x_coordinate = int(voxel_image(0) + 0.5);
		float depth = static_cast<float>(depth_image(image_y_coordinate, image_x_coordinate)) * p.depth_unit_ratio;

		if (depth <= 0.0f) {
			continue;
		}

		float signed_distance_to_voxel_along_camera_ray = depth - ray_distance;

		field(y_field, x_field) = compute_TSDF_value(signed_distance_to_voxel_along_camera_ray, narrow_band_half_width);

	}

	return field;
}

template<typename Scalar>
Eigen::Matrix<Scalar, eig::Dynamic, eig::Dynamic, eig::ColMajor>
Generator<eig::Matrix<Scalar, eig::Dynamic, eig::Dynamic, eig::ColMajor> >::generate__ewa_tsdf_space_inclusive(
			const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic, eig::ColMajor>& depth_image,
			const eig::Matrix<Scalar, 4, 4, eig::ColMajor>& camera_pose,
			int image_y_coordinate){
	throw_assert(false, "Not implemented");
}


}//namespace tsdf


