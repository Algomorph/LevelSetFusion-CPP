/*
 * sdf_gradient_wrt_transformation2d.cpp
 *
 *  Created on: Mar 05, 2019
 *      Author: Fei Shan
 */

//libraries
#include "Eigen/Eigen"

//local
#include "../math/tensor_operations.hpp"
#include "../math/gradients.hpp"
#include "../math/transformation.hpp"


namespace eig = Eigen;

namespace rigid_optimization {
    void gradient_wrt_twist(const eig::MatrixXf& live_field,
                            const eig::Vector3f& twist,
                            const eig::Vector3i& array_offset,
                            const float& voxel_size,
                            eig::Matrix<eig::Vector3f, eig::Dynamic, eig::Dynamic>& gradient_field) {

        math::MatrixXv2f gradient_first_term;
        math::gradient(gradient_first_term, live_field);
        eig::Matrix3f twist_matrix_homo_inv2d = math::transformation_vector_to_matrix2d(-twist);

        float x_voxel, z_voxel, w_voxel = 1;

        for (int y_field=0; y_field<live_field.rows(); ++y_field) {
            for (int x_field=0; x_field<live_field.cols(); ++x_field) {
//                gradient_field(y_field, x_field) = eig::Vector3f(0.0f, 0.0f, 0.0f);

                x_voxel = (x_field + array_offset[0]) * voxel_size; // x coordinate
                z_voxel = (y_field + array_offset[2]) * voxel_size; // z coordinate

                eig::Vector3f point(x_voxel, z_voxel, w_voxel);
                eig::Vector3f trans_point = twist_matrix_homo_inv2d * point;
                eig::Matrix<float, 2, 3> gradient_second_term;
                gradient_second_term << 1, 0, trans_point[1],
                                        0, 1, -trans_point[0];


                eig::Vector3f gradient = eig::Vector2f(gradient_first_term(y_field, x_field)[0],
                                                       gradient_first_term(y_field, x_field)[1]).transpose()
                                         * gradient_second_term;
                gradient_field(y_field, x_field) = gradient/voxel_size;
            }
        }
    };
}

