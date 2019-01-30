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

//local
#include "ewa.hpp"

namespace tsdf {

eig::MatrixXf generate_2d_TSDF_field_from_depth_image_EWA(
		const eig::Matrix<unsigned char, eig::Dynamic, eig::Dynamic>& depth_image, float depth_unit_ratio,
		const eig::Matrix3f& camera_intrinsic_matrix, const eig::Matrix4f& camera_pose,
		const eig::Vector3f& array_offset, int field_size, float voxel_size, int narrow_band_width_voxels) {
	eig::MatrixXf field(field_size, field_size);
	int narrow_band_half_width = narrow_band_width_voxels / 2 * voxel_size;

	float w_voxel = 1.0f;
	float y_voxel = 0.0f;

	eig::Matrix3f camera_rotation_matrix = camera_pose.block(0,0,3,3);
	eig::Matrix3f covariance_voxel_sphere_world_space = eig::MatrixXf::Identity(3,3) * voxel_size;
	eig::Matrix3f covariance_camera_space =
			camera_rotation_matrix * covariance_voxel_sphere_world_space * camera_rotation_matrix.transpose();

	float squared_radius_threshold = 4.0f;
#pragma omp parallel for
	for(int x_field = 0; x_field < field.cols(); x_field ++){
		for (int y_field = 0; y_field < field.rows(); y_field++){
			float x_voxel = (x_field + array_offset(0)) * voxel_size;
			float z_voxel = (y_field + array_offset(2)) * voxel_size;

			eig::Vector4f voxel_world;
			voxel_world << x_voxel, y_voxel, z_voxel, w_voxel;
			eig::Vector3f voxel_camera = (camera_pose * voxel_world).topRows(3);
			if (voxel_camera(2) <= 0.0f){
				continue;
			}
			// ray distance from camera to voxel center
			float ray_distance = voxel_camera.norm();
			// squared distance from camera to voxel center
			float z_cam_squared = voxel_camera(2)*voxel_camera(2);
			float inv_z_cam = 1.0f / z_cam_squared;

			eig::Matrix3f projection_jacobian;
			projection_jacobian <<
					inv_z_cam, 0.0f, -voxel_camera(0) / z_cam_squared,
					0.0f, inv_z_cam, -voxel_camera(1) / z_cam_squared,
					voxel_camera(0) / ray_distance, voxel_camera(1) / ray_distance, voxel_camera(2) / ray_distance;
			eig::Matrix3f remapped_covariance =
					projection_jacobian * covariance_camera_space * projection_jacobian.transpose();



		}
	}

	return field;
}

} // namespace tsdf
