/*
 * sdf_gradient_wrt_transformation_tensor.cpp
 *
 *  Created on: Jun 03, 2019
 *      Author: Fei Shan
 */

// local
#include "sdf_gradient_wrt_transformation_tensor.tpp"

namespace rigid_optimization {

template void gradient_wrt_twist<float>(const eig::Tensor<float, 3>& live_field,
                                 const eig::Matrix<float, 6, 1>& twist,
                                 const eig::Vector3i& array_offset,
                                 const float& voxel_size,
                                 eig::Tensor<eig::Matrix<float, 6, 1>, 3>& gradient_field);

}