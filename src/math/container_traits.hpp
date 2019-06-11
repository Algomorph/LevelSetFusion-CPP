/*
 * get_dimension_count.hpp
 *
 *  Created on: Apr 11, 2019
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

// libraries
#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>

//local
#include "typedefs.hpp"

namespace math {

template<typename Container>
class ContainerWrapper {};

template<typename Scalar>
class ContainerWrapper<Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> >{
public:
	static const int DimensionCount = 2;
	static const int DegreeOfFreedom = 3;
	typedef math::Vector2i Coordinates;
	typedef Eigen::Matrix<math::Vector2<Scalar>,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> VectorContainer;
	typedef Eigen::Matrix<math::Matrix2<Scalar>,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> MatrixContainer;
    typedef Eigen::Matrix<Scalar, 3, 1> TransformationVector;
    typedef Eigen::Matrix<Scalar, 3, 3> TransformationMatrix;
    typedef Eigen::Matrix<Scalar, 3, 3> Sdf2SdfOptimizerCoefficientA;
//    typedef Eigen::Matrix<Scalar, 3, 1> Sdf2SdfOptimizerCoefficientB;
};

template<typename Scalar, int DimensionCountIn>
class ContainerWrapper<Eigen::Tensor<Scalar, DimensionCountIn, Eigen::ColMajor>>{
public:
	static const int DimensionCount = DimensionCountIn;
	typedef math::Vector3i Coordinates;
};

template<typename Scalar>
class ContainerWrapper<Eigen::Tensor<Scalar, 3, Eigen::ColMajor>>{
public:
	static const int DimensionCount = 3;
    static const int DegreeOfFreedom = 6;
	typedef math::Vector3i Coordinates;
	typedef Eigen::Tensor<math::Vector3<Scalar>,3,Eigen::ColMajor> VectorContainer;
	typedef Eigen::Tensor<math::Matrix3<Scalar>,3,Eigen::ColMajor> MatrixContainer;
    typedef Eigen::Matrix<Scalar, 6, 1> TransformationVector;
	typedef Eigen::Matrix<Scalar, 4, 4> TransformationMatrix;
    typedef Eigen::Matrix<Scalar, 6, 6> Sdf2SdfOptimizerCoefficientA;
//    typedef Eigen::Matrix<Scalar, 6, 1> Sdf2SdfOptimizerCoefficientB;
};

} // namespace math
