/*
 * sdf2sdf_optimizer.cpp
 *
 *  Created on: May 31, 2019
 *      Author: Fei Shan
 */

//local
#include "sdf_2_sdf_optimizer.tpp"

#include "../math/typedefs.hpp"

namespace rigid_optimization {

template class Sdf2SdfOptimizer<eig::MatrixXf, eig::Matrix<eig::Matrix<float, 3, 1>, eig::Dynamic, eig::Dynamic>>;
template class Sdf2SdfOptimizer<math::Tensor3f, eig::Tensor<eig::Matrix<float, 6, 1>, 3>>;

}
