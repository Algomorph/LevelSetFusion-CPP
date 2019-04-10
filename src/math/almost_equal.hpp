/*
 * assesment.hpp
 *
 *  Created on: Feb 1, 2019
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
#include "typedefs.hpp"

namespace math {

bool almost_equal(float a, float b);
bool almost_equal(double a, double b);
bool almost_equal(float a, float b, float tolerance);
bool almost_equal(float a, float b, double tolerance);
bool almost_equal(double a, double b, double tolerance);

template<typename ElementType, typename ToleranceType>
bool almost_equal(math::Vector2<ElementType> a, math::Vector2<ElementType> b, ToleranceType tolerance);

template<typename ElementType, typename ToleranceType>
bool almost_equal(math::Vector3<ElementType> a, math::Vector3<ElementType> b, ToleranceType tolerance);

template<typename ElementType, typename ToleranceType>
bool almost_equal(math::Matrix2<ElementType> a, math::Matrix2<ElementType> b, ToleranceType tolerance);

template<typename ElementType, typename ToleranceType>
bool almost_equal(Eigen::Matrix<ElementType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> a,
		Eigen::Matrix<ElementType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> b, ToleranceType tolerance);

template<typename ElementType, typename ToleranceType>
bool almost_equal_verbose(Eigen::Matrix<ElementType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> a,
		Eigen::Matrix<ElementType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> b, ToleranceType tolerance);

template<typename ElementType, typename ToleranceType>
bool almost_equal(Eigen::Tensor<ElementType, 3, Eigen::ColMajor> a,
		Eigen::Tensor<ElementType, 3, Eigen::ColMajor> b, ToleranceType tolerance);

template<typename ElementType, typename ToleranceType>
bool almost_equal_verbose(Eigen::Tensor<ElementType, 3, Eigen::ColMajor> a,
		Eigen::Tensor<ElementType, 3, Eigen::ColMajor> b, ToleranceType tolerance);


//TODO: old stuff, remove
//template<typename TCollection>
//bool almost_equal(TCollection a, TCollection b);
//template<typename TCollection, typename TElementType>
//bool almost_equal(TCollection a, TCollection b, TElementType tolerance);

//TODO: templatize further into almost_equal and almost_equal_verbose (unified data structure API)
//template<typename TMatrix, typename TToleranceType>
//bool matrix_almost_equal(TMatrix matrix_a, TMatrix matrix_b, TToleranceType tolerance = 1e-10);
//template<typename TMatrix, typename TToleranceType>
//bool matrix_almost_equal_verbose(TMatrix matrix_a, TMatrix matrix_b, TToleranceType tolerance = 1e-10);
//template<typename TTensor, typename TToleranceType>
//bool tensor_almost_equal(TTensor container_a, TTensor container_b, TToleranceType tolerance = 1e-10);
//template<typename TTensor, typename TToleranceType>
//bool tensor_almost_equal_verbose(TTensor container_a, TTensor container_b, TToleranceType tolerance = 1e-10);

}//namespace math
