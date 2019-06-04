/*
 * sdf_gradient_wrt_transformation_tensor.tpp
 *
 *  Created on: Jun 03, 2019
 *      Author: Fei Shan
 */

#pragma once

//libraries
#include "Eigen/Eigen"

//local
#include "sdf_gradient_wrt_transformation.hpp"
#include "../math/gradients.hpp"
#include "../math/transformation.hpp"


namespace eig = Eigen;

namespace rigid_optimization {

template<typename Scalar>
void gradient_wrt_twist(const eig::Tensor<Scalar, 3>& live_field,
                        const eig::Matrix<float, 6, 1>& twist,
                        const eig::Vector3i& array_offset,
                        const float& voxel_size,
                        eig::Tensor<eig::Matrix<Scalar, 6, 1>, 3>& gradient_field){

    eig::Tensor<math::Vector3<Scalar>, 3> gradient_first_term;
    math::gradient(gradient_first_term, live_field);
    eig::Matrix<float, 6, 1> inv_twist = -twist;
    eig::Matrix4f inv_twist_matrix3d = math::transformation_vector_to_matrix(inv_twist);

    float x_voxel, y_voxel, z_voxel, w_voxel = 1;

    int x_size = live_field.dymention(0);
    int y_size = live_field.dymention(1);
//    int z_size = live_field.dymention(2);

    int y_stride = x_size;
    int z_stride = y_stride * y_size;

    int voxel_count = static_cast<int>(live_field.size());

#pragma omp parallel for
    for (int i_element = 0; i_element < voxel_count; i_element++) {
        int z_field = i_element / z_stride;
        int remainder = i_element % z_stride;
        int y_field = remainder / y_stride;
        int x_field = remainder % y_stride;

        x_voxel = (x_field + array_offset[0]) * voxel_size; // x coordinate
        y_voxel = (y_field + array_offset[1]) * voxel_size; // y coordinate
        z_voxel = (z_field + array_offset[2]) * voxel_size; // z coordinate

        eig::Vector4f point(x_voxel, y_voxel, z_voxel, w_voxel);
        eig::Vector4f trans_point = inv_twist_matrix3d * point;
        eig::Matrix<Scalar, 3, 6> gradient_second_term;
        gradient_second_term << Scalar(1.f), Scalar(0.f), Scalar(0.f), Scalar(0.f), Scalar(trans_point[2]), -Scalar(trans_point[1]),
                                Scalar(0.f), Scalar(1.f), Scalar(0.f), -Scalar(trans_point[2]), Scalar(0.f), Scalar(trans_point[0]),
                                Scalar(0.f), Scalar(0.f), Scalar(1.f), Scalar(trans_point[1]), -Scalar(trans_point[0], Scalar(0.f));
        eig::Matrix<Scalar, 6, 1> gradient = eig::Vector3f(gradient_first_term(x_field, y_field, z_field)[0],
                                                          gradient_first_term(x_field, y_field, z_field)[1],
                                                          gradient_first_term(x_field, y_field, z_field)[2]).transpose()
                                             * gradient_second_term;
        gradient_field(x_field, y_field, z_field) = gradient/voxel_size;
    }
}
;

}

