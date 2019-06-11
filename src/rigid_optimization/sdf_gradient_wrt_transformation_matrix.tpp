/*
 * sdf_gradient_wrt_transformation_matrix.tpp
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

template <typename Scalar, typename Coordinates>
void gradient_wrt_twist(const eig::Matrix<Scalar, eig::Dynamic, eig::Dynamic>& live_field,
                        const eig::Matrix<Scalar, 3, 1>& twist,
                        const Coordinates& array_offset,
                        const Scalar& voxel_size,
                        const eig::Matrix<Scalar, eig::Dynamic, eig::Dynamic>& canonical_field, // canonical_field is only used to calculate vector_b.
                        eig::Matrix<eig::Matrix<Scalar, 3, 1>, eig::Dynamic, eig::Dynamic>& gradient_field, // gradient_field is the gradient of live_field.
                        eig::Matrix<Scalar, 3, 3>& matrix_A,
                        eig::Matrix<Scalar, 3, 1>& vector_b) {

    int x_size = live_field.cols();
    int y_size = live_field.rows();

    eig::Matrix<math::Vector2<Scalar>, Eigen::Dynamic, Eigen::Dynamic> gradient_first_term(y_size, x_size);
    math::gradient(gradient_first_term, live_field);
    eig::Vector3f inv_twist = -twist;
    eig::Matrix3f inv_twist_matrix2d = math::transformation_vector_to_matrix(inv_twist);

    float x_voxel, z_voxel, w_voxel = 1.f;

    int matrix_size = static_cast<int>(live_field.size());

    for (int i_element = 0; i_element < matrix_size; i_element++) {
        // Any MatrixXf in Eigen is column-major
        // i_element = x * column_count + y
        int x_field = i_element / x_size;
        int y_field = i_element % x_size;

        x_voxel = (x_field + array_offset[0]) * voxel_size; // x coordinate
        z_voxel = (y_field + array_offset[1]) * voxel_size; // z coordinate

        eig::Vector3f point(x_voxel, z_voxel, w_voxel);
        eig::Vector3f trans_point = inv_twist_matrix2d * point;
        eig::Matrix<Scalar, 2, 3> gradient_second_term;
        gradient_second_term << Scalar(1.f), Scalar(0.f), Scalar(trans_point[1]),
                                Scalar(0.f), Scalar(1.f), Scalar(-trans_point[0]);

        eig::Matrix<Scalar, 3, 1> gradient = eig::Matrix<Scalar, 2, 1>(gradient_first_term(y_field, x_field)[0],
                                                                       gradient_first_term(y_field, x_field)[1]).transpose()
                                             * gradient_second_term;
        gradient_field(y_field, x_field) = gradient/voxel_size;

        matrix_A += gradient_field(y_field, x_field) * gradient_field(y_field, x_field).transpose();

        vector_b += (canonical_field(y_field, x_field) - live_field(y_field, x_field) + gradient_field(y_field, x_field).transpose() * twist)
                    * gradient_field(y_field, x_field);
    }
}
;

template <typename Scalar>
eig::Matrix<eig::Matrix<Scalar, 3, 1>, eig::Dynamic, eig::Dynamic> init_gradient_wrt_twist(const eig::Matrix<Scalar, eig::Dynamic, eig::Dynamic>& live_field){
    eig::Matrix<eig::Matrix<Scalar, 3, 1>, eig::Dynamic, eig::Dynamic> gradient(live_field.rows(),
                                                                                live_field.cols());
    gradient.setZero();
    return gradient;
}
;

}


