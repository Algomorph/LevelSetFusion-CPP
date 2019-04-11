/*
 * field_like.tpp
 *
 *  Created on: Apr 10, 2019
 *      Author: Gregory Kramida
 *   Copyright: 2019 Gregory Kramida
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */

#pragma once

//libraries
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

//local
#include "field_like.hpp"

namespace math{

template<typename Scalar, typename InField>
static inline
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
scalar_field_like_matrix_aux(const InField& field){
	return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>(field.rows(), field.cols());
}

template<typename Scalar, typename InField>
static inline
Eigen::Tensor<Scalar, 3, Eigen::ColMajor>
scalar_field_like_tensor_aux(const InField& field){
	return Eigen::Tensor<Scalar, 3, Eigen::ColMajor>(field.dimensions());
}

template<typename Scalar, typename InField>
static inline
Eigen::Matrix<math::Vector2<Scalar>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
vector_field_like_matrix_aux(const InField& field){
	return Eigen::Matrix<math::Vector2<Scalar>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>(field.rows(), field.cols());
}

template<typename Scalar, typename InField>
static inline
Eigen::Tensor<math::Vector3<Scalar>, 3, Eigen::ColMajor>
vector_field_like_tensor_aux(const InField& field){
	return Eigen::Tensor<math::Vector3<Scalar>, 3, Eigen::ColMajor>(field.dimensions());
}


template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
scalar_field_like(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& field){
	return scalar_field_like_matrix_aux<Scalar, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >(field);
}

template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
scalar_field_like(const Eigen::Matrix<math::Vector2<Scalar>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& field){
	return scalar_field_like_matrix_aux<Scalar, Eigen::Matrix<math::Vector2<Scalar>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >(field);
}

template<typename Scalar>
Eigen::Tensor<Scalar, 3, Eigen::ColMajor>
scalar_field_like(const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& field){
	return scalar_field_like_tensor_aux<Scalar,Eigen::Tensor<Scalar, 3, Eigen::ColMajor> >(field);
}

template<typename Scalar>
Eigen::Matrix<math::Vector2<Scalar>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
vector_field_like(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& field){
	return vector_field_like_matrix_aux<Scalar, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >(field);
}

template<typename Scalar>
Eigen::Matrix<math::Vector2<Scalar>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
vector_field_like(const Eigen::Matrix<math::Vector2<Scalar>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& field){
	return vector_field_like_matrix_aux<Scalar, Eigen::Matrix<math::Vector2<Scalar>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >(field);
}

template<typename Scalar>
Eigen::Tensor<math::Vector3<Scalar>, 3, Eigen::ColMajor>
vector_field_like(const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& field){
	return vector_field_like_tensor_aux<Scalar, Eigen::Tensor<Scalar, 3, Eigen::ColMajor> >(field);
}

template<typename Scalar>
Eigen::Tensor<math::Vector3<Scalar>, 3, Eigen::ColMajor>
vector_field_like(const Eigen::Tensor<math::Vector3<Scalar>, 3, Eigen::ColMajor>& field){
	return vector_field_like_tensor_aux<Scalar, Eigen::Tensor<math::Vector3<Scalar>, 3, Eigen::ColMajor> >(field);
}


} //namespace math


