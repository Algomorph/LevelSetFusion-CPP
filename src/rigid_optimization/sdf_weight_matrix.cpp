/*
 * sdf_weight_matrix.cpp
 *
 *  Created on: Jun 04, 2019
 *      Author: Fei Shan
 */

#include "sdf_weight_matrix.tpp"

namespace eig = Eigen;

namespace rigid_optimization{

template eig::Matrix<float, eig::Dynamic, eig::Dynamic, eig::ColMajor> sdf_weight<float>(const eig::Matrix<float, eig::Dynamic, eig::Dynamic, eig::ColMajor>& field, const float& eta);

}


