/*
 * sdf_gradient_wrt_transformation2d.hpp
 *
 *  Created on: Jun 03, 2019
 *      Author: Fei Shan
 */

//libraries
#include "Eigen/Eigen"

namespace eig = Eigen;

namespace rigid_optimization {

template <typename Scalar>
void gradient_wrt_twist(const eig::Matrix<Scalar, eig::Dynamic, eig::Dynamic>& live_field,
                        const eig::Vector3f& twist,
                        const eig::Vector3i& array_offset,
                        const float& voxel_size,
                        eig::Matrix<eig::Matrix<Scalar, 3, 1>, eig::Dynamic, eig::Dynamic>& gradient_field);

template<typename Scalar>
void gradient_wrt_twist(const eig::Tensor<Scalar, 3>& live_field,
                        const eig::Matrix<float, 6, 1>& twist,
                        const eig::Vector3i& array_offset,
                        const float& voxel_size,
                        eig::Tensor<eig::Matrix<Scalar, 6, 1>, 3>& gradient_field);
}