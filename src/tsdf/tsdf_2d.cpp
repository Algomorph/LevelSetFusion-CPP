//  =================================================7===============
//  Created by Fei Shan on 03/19/19.
//  ================================================================




//local
#include "tsdf.hpp"
#include "common.hpp"

namespace tsdf {

eig::MatrixXf generate_TSDF_2D(
        int image_y_coordinate,
        const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic> &depth_image,
        float depth_unit_ratio,
        const eig::Matrix3f &camera_intrinsic_matrix,
        const eig::Matrix4f &camera_pose,
        const eig::Vector3i &array_offset,
        int field_size,
        float voxel_size,
        int narrow_band_width_voxels,
        float default_value) {
    eig::MatrixXf field(field_size, field_size);
    std::fill_n(field.data(), field.size(), default_value);
    float narrow_band_half_width = static_cast<float>(narrow_band_width_voxels / 2) * voxel_size;

    float w_voxel = 1.0f;
    float y_voxel = 0.0f;

    int matrix_size = static_cast<int>(field.size());

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

        if (is_voxel_out_of_bounds(voxel_image, depth_image, 0)) {
            continue;
        }

        // ray distance from camera to voxel center
        // TODO: there is difference between voxel_camera.norm() and voxel_camera[2].
        float ray_distance = voxel_camera(2);

        int image_x_coordinate = int(voxel_image(0) + 0.5);
        float depth = static_cast<float>(depth_image(image_y_coordinate, image_x_coordinate)) * depth_unit_ratio;

        if (depth <= 0.0f) {
            continue;
        }

        float signed_distance_to_voxel_along_camera_ray = depth - ray_distance;

        field(y_field, x_field) = compute_TSDF_value(signed_distance_to_voxel_along_camera_ray, narrow_band_half_width);

    }

    return field;
}
};
