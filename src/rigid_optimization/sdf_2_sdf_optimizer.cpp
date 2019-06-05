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

template class Sdf2SdfOptimizer<float, eig::MatrixXf, tsdf::Parameters2d, tsdf::Generator2d, eig::Matrix3f>;
template class Sdf2SdfOptimizer<float, math::Tensor3f, tsdf::Parameters3d, tsdf::Generator3d, eig::Matrix4f>;

}
