/*
 * sdf_weight.hpp
 *
 *  Created on: Jun 04, 2019
 *      Author: Fei Shan
 */

#pragma once

//libraries
#include <Eigen/Eigen>

//local
#include "../math/typedefs.hpp"

namespace eig = Eigen;

namespace rigid_optimization {

template<typename Scalar>
eig::Matrix<Scalar, eig::Dynamic, eig::Dynamic, eig::ColMajor> sdf_weight(const eig::Matrix<Scalar, eig::Dynamic, eig::Dynamic, eig::ColMajor>& field, const Scalar& eta);

template<typename Scalar>
eig::Tensor<Scalar, 3, eig::ColMajor> sdf_weight(const eig::Tensor<Scalar, 3, eig::ColMajor>& field, const Scalar& eta);

}
