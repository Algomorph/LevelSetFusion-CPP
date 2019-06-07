/*
 * sdf_gradient_wrt_transformation2d.hpp
 *
 *  Created on: Jun 03, 2019
 *      Author: Fei Shan
 */

//libraries
#include "Eigen/Eigen"
#include <unsupported/Eigen/CXX11/Tensor>

namespace eig = Eigen;

namespace rigid_optimization {

template <typename Scalar>
void gradient_wrt_twist(const eig::Matrix<Scalar, eig::Dynamic, eig::Dynamic>& live_field,
                        const eig::Vector3f& twist,
                        const eig::Vector3i& array_offset,
                        float voxel_size,
                        const eig::Matrix<Scalar, eig::Dynamic, eig::Dynamic>& canonical_field, // canonical_field is only used to calculate vector_b.
                        eig::Matrix<eig::Matrix<Scalar, 3, 1>, eig::Dynamic, eig::Dynamic>& gradient_field, // gradient_field is the gradient of live_field.
                        eig::Matrix<Scalar, 3, 3>& matrix_A,
                        eig::Matrix<Scalar, 3, 1>& vector_b);

template<typename Scalar>
void gradient_wrt_twist(const eig::Tensor<Scalar, 3>& live_field,
                        const eig::Matrix<float, 6, 1>& twist,
                        const eig::Vector3i& array_offset,
                        float voxel_size,
                        const eig::Tensor<Scalar, 3>& canonical_field, // canonical_field is only used to calculate vector_b.
                        eig::Tensor<eig::Matrix<Scalar, 6, 1>, 3>& gradient_field, // gradient_field is the gradient of live_field.
                        eig::Matrix<Scalar, 6, 6>& matrix_A,
                        eig::Matrix<Scalar, 6, 1>& vector_b);
}