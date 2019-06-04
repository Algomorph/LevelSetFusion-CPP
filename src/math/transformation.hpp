//  ================================================================
//  Created by Fei Shan on 03/05/19.
//  Converting transformation vector to matrix in 2D and 3D homogeneous coordinates.
//  ================================================================

#pragma once

//libraries
#include <Eigen/Eigen>

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
eig::Matrix<Scalar, 4, 4> transformation_vector_to_matrix(const eig::Matrix<Scalar, 6, 1>& twist);

}
