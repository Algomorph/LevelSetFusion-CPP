//  ================================================================
//  Created by Fei Shan on 03/05/19.
//  Converting transformation vector to matrix in 2D and 3D homogeneous coordinates.
//  ================================================================

#include "transformation.tpp"

namespace math {

template eig::Matrix3f transformation_vector_to_matrix<float>(const eig::Vector3f& twist);
template eig::Matrix4f transformation_vector_to_matrix<float>(const eig::Matrix<float, 6, 1>& twist);

template eig::Matrix4f inverse_transformation_matrix<float>(const eig::Matrix<float, 4, 4>& twist);

template eig::Matrix3f init_transformation_matrix<float>(const eig::MatrixXf& field);
template eig::Matrix4f init_transformation_matrix<float>(const eig::Tensor<float, 3>& field);

template eig::Vector3f init_transformation_vector<float>(const eig::MatrixXf& field);
template eig::Matrix<float, 6, 1> init_transformation_vector<float>(const eig::Tensor<float, 3>& field);

template eig::Matrix<float, 6, 1> to_3d_transformation_vector<float>(const eig::Vector3f& twist);
template eig::Matrix<float, 6, 1> to_3d_transformation_vector<float>(const eig::Matrix<float, 6, 1>& twist); // Stay the same

template eig::Matrix<float, 3, 1> to_2d_transformation_vector<float>(const eig::Matrix<float, 6, 1>& twist);
template eig::Matrix<float, 3, 1> to_2d_transformation_vector<float>(const eig::Vector3f& twist); // Stay the same

}

