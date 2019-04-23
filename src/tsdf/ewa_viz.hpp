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
#include "../math/typedefs.hpp"

namespace eig = Eigen;

namespace tsdf {


math::MatrixXuc generate_TSDF_3D_EWA_image_visualization(
		const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& depth_image,
		float depth_unit_ratio,
		const eig::Tensor<float, 3>& field,
		const eig::Matrix3f& camera_intrinsic_matrix,
		const eig::Matrix4f& camera_pose = eig::Matrix4f::Identity(4, 4),
		const eig::Vector3i& array_offset =
				[] {eig::Vector3i default_offset; default_offset << -64, -64, 64; return default_offset;}(),
		float voxel_size = 0.004,
		int scale=20,
		float tsdf_threshold = 0.1f,
		float gaussian_covariance_scale = 1.0f);

eig::MatrixXf sampling_area_heatmap_2D_EWA_image(int image_y_coordinate,
		const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& depth_image,
		float depth_unit_ratio,
		const eig::Matrix3f& camera_intrinsic_matrix,
		const eig::Matrix4f& camera_pose,
		const eig::Vector3i& array_offset,
		int field_size,
		float voxel_size,
		int narrow_band_width_voxels,
		float gaussian_covariance_scale);

} // namespace tsdf
