//  ================================================================
//  Created by Fei Shan on 03/05/19.
//  Converting transformation vector to matrix in 2D and 3D homogeneous coordinates.
//  ================================================================

#pragma once

//libraries
#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>

namespace eig = Eigen;

namespace math{

/**
 * @brief Transform the transformation vector to corresponding transformation matrix in 2D.
 * @details
 * @param[out] 3 by 3 matrix output
 * @param[in] 3 by 1 vector input
 */
template<typename Scalar>
eig::Matrix<Scalar, 3, 3> transformation_vector_to_matrix(const eig::Matrix<Scalar, 3, 1>& twist);

/**
 * @brief Transform the transformation vector to corresponding transformation matrix in 3D.
 * @details
 * @param[out] 4 by 4 matrix output
 * @param[in] 6 by 1 vector input
 */

template<typename Scalar>
eig::Matrix<Scalar, 4, 4> inverse_transformation_matrix(const eig::Matrix<Scalar, 4, 4>& twist_matrix);

template<typename Scalar>
eig::Matrix<Scalar, 4, 4> transformation_vector_to_matrix(const eig::Matrix<Scalar, 6, 1>& twist);

template <typename Scalar>
eig::Matrix<Scalar, 3, 3> init_transformation_matrix(const eig::Matrix<Scalar, eig::Dynamic, eig::Dynamic>& field);

template <typename Scalar>
eig::Matrix<Scalar, 4, 4> init_transformation_matrix(const eig::Tensor<Scalar, 3>& field);

template <typename Scalar>
eig::Matrix<Scalar, 3, 1> init_transformation_vector(const eig::Matrix<Scalar, eig::Dynamic, eig::Dynamic>& field);

template <typename Scalar>
eig::Matrix<Scalar, 6, 1> init_transformation_vector(const eig::Tensor<Scalar, 3>& field);

template <typename Scalar>
eig::Matrix<Scalar, 6, 1> to_3d_transformation_vector(const eig::Matrix<Scalar, 3, 1>& twist);

template <typename Scalar>
eig::Matrix<Scalar, 6, 1> to_3d_transformation_vector(const eig::Matrix<Scalar, 6, 1>& twist); // Stay the same

template <typename Scalar>
eig::Matrix<Scalar, 3, 1> to_2d_transformation_vector(const eig::Matrix<Scalar, 6, 1>& twist);

template <typename Scalar>
eig::Matrix<Scalar, 3, 1> to_2d_transformation_vector(const eig::Matrix<Scalar, 3, 1>& twist); // Stay the same
}
