/*
 * sdf_weight_tensor.cpp
 *
 *  Created on: Jun 04, 2019
 *      Author: Fei Shan
 */

#include "sdf_weight_tensor.tpp"

namespace eig = Eigen;

namespace rigid_optimization{

template eig::Tensor<float, 3, eig::ColMajor> sdf_weight<float>(const eig::Tensor<float, 3, eig::ColMajor>& field, float eta);

}
