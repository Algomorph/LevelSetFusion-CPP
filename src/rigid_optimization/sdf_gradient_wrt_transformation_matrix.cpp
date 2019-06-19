/*
 * sdf_gradient_wrt_transformation_matrix.cpp
 *
 *  Created on: Jun 03, 2019
 *      Author: Fei Shan
 */

// local
#include "sdf_gradient_wrt_transformation_matrix.tpp"

namespace rigid_optimization {

template float gradient_wrt_twist<float, math::Vector2i>(const eig::MatrixXf& live_field,
                                                         const eig::Vector3f& twist,
                                                         const math::Vector2i& array_offset,
                                                         const float& voxel_size,
                                                         const eig::MatrixXf& canonical_field, // canonical_field is only used to calculate vector_b.
                                                         eig::Matrix<eig::Vector3f, eig::Dynamic, eig::Dynamic>& gradient_field, // gradient_field is the gradient of live_field.
                                                         eig::Matrix3f& matrix_A,
                                                         eig::Vector3f& vector_b);

template eig::Matrix<eig::Matrix<float, 3, 1>, eig::Dynamic, eig::Dynamic> init_gradient_wrt_twist<float>(const eig::Matrix<float, eig::Dynamic, eig::Dynamic>& live_field);

}