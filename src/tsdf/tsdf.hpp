//  ================================================================
//  Created by Fei Shan on 03/19/19.
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

//constexpr float near_clipping_distance = 0.05; //m

eig::MatrixXf generate_TSDF_2D( // no interpolation
        int image_y_coordinate,
        const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& depth_image,
        float depth_unit_ratio,
        const eig::Matrix3f& camera_intrinsic_matrix,
        const eig::Matrix4f& camera_pose = eig::Matrix4f::Identity(4, 4),
        const eig::Vector3i& array_offset =
        [] {eig::Vector3i default_offset; default_offset << -64, -64, 64; return default_offset;}(),
        int field_size = 128,
        float voxel_size = 0.004,
        int narrow_band_width_voxels = 20,
        float default_value = 1.0f);
};
