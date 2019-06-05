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

template void export_algorithms<float, eig::MatrixXf, tsdf::Parameters2d, tsdf::Generator2d, eig::Matrix3f>(const char* suffix);
template void export_algorithms<float, math::Tensor3f, tsdf::Parameters3d, tsdf::Generator3d, eig::Matrix4f>(const char* suffix);

}
}
