//  ================================================================
//  Created by Gregory Kramida on 10/11/18.
//  Copyright (c) 2018 Gregory Kramida
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//  ================================================================
#pragma once
//stdlib

//libraries
#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>
#include <boost/python.hpp>

//local
#include "../math/typedefs.hpp"

namespace bp = boost::python;
namespace eig = Eigen;

namespace nonrigid_optimization {

template<typename ElementType, typename Scalar>
Eigen::Matrix<ElementType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
warp(const Eigen::Matrix<ElementType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& scalar_field,
	 const Eigen::Matrix<math::Vector2<Scalar>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& warp_field);

template<typename ElementType, typename Scalar>
eig::Tensor<ElementType, 3, Eigen::ColMajor>
warp(const eig::Tensor<ElementType, 3, Eigen::ColMajor>& scalar_field,
	 const eig::Tensor<math::Vector3<Scalar>, 3, Eigen::ColMajor>& warp_field);

template<typename ElementType, typename Scalar>
Eigen::Matrix<ElementType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
warp_with_replacement(const Eigen::Matrix<ElementType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& scalar_field,
				      const Eigen::Matrix<math::Vector2<Scalar>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& warp_field,
				      ElementType replacement_value);

template<typename ElementType, typename Scalar>
eig::Tensor<ElementType, 3, Eigen::ColMajor>
warp_with_replacement(const eig::Tensor<ElementType, 3, Eigen::ColMajor>& scalar_field,
					  const eig::Tensor<math::Vector3<Scalar>, 3, Eigen::ColMajor>& warp_field,
					  ElementType replacement_value);

//================= legacy (Sobolev/KillingFusion only) ========================
eig::MatrixXf warp_2d_advanced(math::MatrixXv2f& warp_field,
		const eig::MatrixXf& warped_live_field, const eig::MatrixXf& canonical_field,
		bool band_union_only = false, bool known_values_only = false,
		bool substitute_original = false, float truncation_float_threshold = 1e-6);
eig::MatrixXf warp_2d_advanced_warp_unchanged(math::MatrixXv2f& warp_field,
		const eig::MatrixXf& warped_live_field, const eig::MatrixXf& canonical_field,
		bool band_union_only = false, bool known_values_only = false,
		bool substitute_original = false, float truncation_float_threshold = 1e-6);


bp::object py_warp_field(const eig::MatrixXf& warped_live_field,
		const eig::MatrixXf& canonical_field, eig::MatrixXf warp_field_u,
		eig::MatrixXf warp_field_v, bool band_union_only = false, bool known_values_only = false,
		bool substitute_original = false, float truncation_float_threshold = 1e-6);

bp::object py_warp_field_no_warp_change(const eig::MatrixXf& warped_live_field,
		const eig::MatrixXf& canonical_field, eig::MatrixXf warp_field_u,
		eig::MatrixXf warp_field_v, bool band_union_only = false, bool known_values_only = false,
		bool substitute_original = false, float truncation_float_threshold = 1e-6);

} // namespace nonrigid_optimization

