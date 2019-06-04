//  ================================================================
//  Created by Fei Shan on 03/05/19.
//  Converting transformation vector to matrix in 2D and 3D homogeneous coordinates.
//  ================================================================

#include "transformation.tpp"

namespace math {

template eig::Matrix3f transformation_vector_to_matrix<float>(const eig::Vector3f& twist);
template eig::Matrix4f transformation_vector_to_matrix<float>(const eig::Matrix<float, 6, 1>& twist);

}

