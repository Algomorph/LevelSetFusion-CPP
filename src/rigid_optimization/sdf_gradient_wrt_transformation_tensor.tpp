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

template<typename Scalar, typename Coordinates>
float gradient_wrt_twist(const eig::Tensor<Scalar, 3>& live_field,
                         const eig::Matrix<Scalar, 6, 1>& twist,
                         const Coordinates& array_offset,
                         const Scalar& voxel_size,
                         const eig::Tensor<Scalar, 3>& canonical_field, // canonical_field is only used to calculate vector_b.
                         eig::Tensor<eig::Matrix<Scalar, 6, 1>, 3>& gradient_field, // gradient_field is the gradient of live_field.
                         eig::Matrix<Scalar, 6, 6>& matrix_A,
                         eig::Matrix<Scalar, 6, 1>& vector_b) {

    int x_size = live_field.dimension(0);
    int y_size = live_field.dimension(1);
    int z_size = live_field.dimension(2);

    eig::Tensor<math::Vector3<Scalar>, 3> gradient_first_term_field(x_size, y_size, z_size);
    math::gradient(gradient_first_term_field, live_field);

    eig::Matrix4f temp_twist_matrix3d = math::transformation_vector_to_matrix(twist);
    eig::Matrix4f inv_twist_matrix3d = math::inverse_transformation_matrix(temp_twist_matrix3d);

    float x_voxel, y_voxel, z_voxel, w_voxel = 1.f;

    int y_stride = x_size;
    int z_stride = y_stride * y_size;

    int voxel_count = static_cast<int>(live_field.size());

    float energy = 0;
    int valid = 0;

    for (int i_element = 0; i_element < voxel_count; i_element++) {
        int z_field = i_element / z_stride;
        int remainder = i_element % z_stride;
        int y_field = remainder / y_stride;
        int x_field = remainder % y_stride;
        // TODO: add 0.5?
        x_voxel = (x_field + array_offset[0]) * voxel_size; // x coordinate
        y_voxel = (y_field + array_offset[1]) * voxel_size; // y coordinate
        z_voxel = (z_field + array_offset[2]) * voxel_size; // z coordinate

        eig::Vector4f point(x_voxel, y_voxel, z_voxel, w_voxel);
        eig::Vector4f trans_point = inv_twist_matrix3d * point;
        eig::Matrix<Scalar, 3, 6> gradient_second_term;
        gradient_second_term << Scalar(1.f), Scalar(0.f), Scalar(0.f), Scalar(0.f), Scalar(trans_point[2]), -Scalar(trans_point[1]),
                                Scalar(0.f), Scalar(1.f), Scalar(0.f), -Scalar(trans_point[2]), Scalar(0.f), Scalar(trans_point[0]),
                                Scalar(0.f), Scalar(0.f), Scalar(1.f), Scalar(trans_point[1]), -Scalar(trans_point[0]), Scalar(0.f);
        eig::Matrix<Scalar, 1, 3> gradient_first_term = eig::Matrix<Scalar, 1, 3>(gradient_first_term_field(x_field, y_field, z_field)[0],
                                                                                  gradient_first_term_field(x_field, y_field, z_field)[1],
                                                                                  gradient_first_term_field(x_field, y_field, z_field)[2])/voxel_size;
        eig::Matrix<Scalar, 6, 1> gradient = gradient_first_term * gradient_second_term;

        gradient_field(x_field, y_field, z_field) = gradient;

        matrix_A += gradient_field(x_field, y_field, z_field) * gradient_field(x_field, y_field, z_field).transpose();

        vector_b += (canonical_field(x_field, y_field, z_field) - live_field(x_field, y_field, z_field) + gradient_field(x_field, y_field, z_field).transpose() * twist)
                    * gradient_field(x_field, y_field, z_field);

        float voxel_wise_difference = canonical_field(x_field, y_field, z_field) * (canonical_field(x_field, y_field, z_field) > -0.01)
                                      - live_field(x_field, y_field, z_field) * (live_field(x_field, y_field, z_field) > -0.01);

        energy += 0.5f * voxel_wise_difference * voxel_wise_difference;

        if (voxel_wise_difference * voxel_wise_difference != 0) {
            valid++;
//            std::cout << canonical_field(x_field, y_field, z_field) << std::endl;
        }

//        if (y_field == 22 && z_field == 12) {
//            std::cout << "x: " << x_field << " SDF " << live_field(x_field, y_field, z_field) << std::endl;
//        }

//        if (x_field == 12 && y_field == 22 && z_field == 12) {
//            std::cout << "voxel_index (" << x_field << " " << y_field << " " << z_field << ")" << std::endl;
//            std::cout << "voxel_center (" << x_voxel << " " << y_voxel << " " << z_voxel << ")" << std::endl;
//            std::cout << "canonical_field " << canonical_field(x_field, y_field, z_field) << std::endl;
//            std::cout << "live_field " <<live_field(x_field, y_field, z_field) << std::endl;
//            std::cout << "gradient_first_term\n" << gradient_first_term << std::endl;
//            std::cout << "gradient_second_term\n"<< gradient_second_term << std::endl;
//            std::cout << "gradient\n"
//                      << gradient_field(x_field, y_field, z_field)[0] << " "
//                      << gradient_field(x_field, y_field, z_field)[1] << " "
//                      << gradient_field(x_field, y_field, z_field)[2] << " "
//                      << gradient_field(x_field, y_field, z_field)[3] << " "
//                      << gradient_field(x_field, y_field, z_field)[4] << " "
//                      << gradient_field(x_field, y_field, z_field)[5] << std::endl;
//
//        }

    }
    std::cout << "valid voxel count: "<< valid << std::endl;

    return energy;
}
;

template <typename Scalar>
eig::Tensor<eig::Matrix<Scalar, 6, 1>, 3> init_gradient_wrt_twist(const eig::Tensor<Scalar, 3>& live_field){
    eig::Tensor<eig::Matrix<Scalar, 6, 1>, 3> gradient(live_field.dimension(0),
                                                       live_field.dimension(1),
                                                       live_field.dimension(2));
    return gradient;
}
;

}


