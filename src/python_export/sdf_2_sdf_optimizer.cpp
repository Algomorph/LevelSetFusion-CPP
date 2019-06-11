/*
 * sdf_2_sdf_optimizer.cpp
 *
 *  Created on: Mar 21, 2019
 *      Author: Fei Shan
 */

#include "sdf_2_sdf_optimizer.tpp"

namespace bp = boost::python;
namespace ro = rigid_optimization;

namespace python_export {
namespace sdf_2_sdf_optimizer {

template void export_algorithms<eig::MatrixXf, eig::Matrix<eig::Matrix<float, 3, 1>, eig::Dynamic, eig::Dynamic>>(const char* suffix);
template void export_algorithms<math::Tensor3f, eig::Tensor<eig::Matrix<float, 6, 1>, 3>>(const char* suffix);

}
}
