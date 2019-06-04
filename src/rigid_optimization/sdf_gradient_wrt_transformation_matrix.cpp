/*
 * sdf_gradient_wrt_transformation_matrix.cpp
 *
 *  Created on: Jun 03, 2019
 *      Author: Fei Shan
 */

// local
#include "sdf_gradient_wrt_transformation_matrix.tpp"

namespace rigid_optimization {

template void gradient_wrt_twist<float> (
        const eig::MatrixXf& live_field,
        const eig::Vector3f& twist,
        const eig::Vector3i& array_offset,
        const float& voxel_size,
        const eig::MatrixXf& canonical_field,
        eig::Matrix<eig::Vector3f, eig::Dynamic, eig::Dynamic>& gradient_field,
        eig::Matrix3f& matrix_A,
        eig::Vector3f& vector_b);

}