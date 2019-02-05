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
#pragma once

//stdlib

//libs
#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>

//local

namespace eig = Eigen;

namespace tsdf {

//eig::MatrixXf generate_3d_TSDF_field_from_depth_image_EWA(
//		const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& depth_image,
//		float depth_unit_ratio,
//		const eig::Matrix3f& camera_intrinsic_matrix,
//		const eig::Matrix4f& camera_pose = eig::Matrix4f::Identity(4,4),
//		const eig::Vector3i& array_offset =
//				[] {eig::Vector3i default_offset; default_offset << -64, -64, 64; return default_offset;}(),
//		int field_size = 128,
//		float voxel_size = 0.004,
//		int narrow_band_width_voxels = 20);

eig::MatrixXf generate_2d_TSDF_field_from_depth_image_EWA(
		int image_y_coordinate,
		const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& depth_image,
		float depth_unit_ratio,
		const eig::Matrix3f& camera_intrinsic_matrix,
		const eig::Matrix4f& camera_pose = eig::Matrix4f::Identity(4,4),
		const eig::Vector3i& array_offset =
				[] {eig::Vector3i default_offset; default_offset << -64, -64, 64; return default_offset;}(),
		int field_size = 128,
		float voxel_size = 0.004,
		int narrow_band_width_voxels = 20);


} // namespace tsdf
