//  ================================================================
//  Created by Gregory Kramida on 10/23/18.
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
#include <cstdlib>

//local
#include "vector2.hpp"
#include "matrix2.hpp"

//libraries
#include <Eigen/Eigen>
#include <iostream>

namespace math {
typedef Eigen::Matrix<math::Vector2<float>, Eigen::Dynamic, Eigen::Dynamic> MatrixXv2f;
typedef Eigen::Matrix<math::Vector2<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXv2f_rm;
typedef Eigen::Matrix<math::Matrix2<float>, Eigen::Dynamic, Eigen::Dynamic> MatrixXm2f;

MatrixXv2f stack_as_xv2f(const Eigen::MatrixXf& matrix_a, const Eigen::MatrixXf& matrix_b);
void unstack_xv2f(Eigen::MatrixXf& matrix_a, Eigen::MatrixXf& matrix_b, const MatrixXv2f vector_field);

template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> upsampleX2(
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& matrix) {
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> upsampled(matrix.rows() * 2, matrix.cols() * 2);
	for (int dest_col = 0, source_col = 0; source_col < matrix.cols(); dest_col += 2, source_col++) {
		for (int dest_row = 0, source_row = 0; source_row < matrix.rows(); dest_row += 2, source_row++) {
			Scalar value = matrix(source_row, source_col);
			upsampled(dest_row, dest_col) = value;
			upsampled(dest_row + 1, dest_col) = value;
			upsampled(dest_row, dest_col + 1) = value;
			upsampled(dest_row + 1, dest_col + 1) = value;
		}
	}
	return upsampled;
}

} //namespace math

namespace Eigen {

template<>
struct NumTraits<math::Vector2<float>>
:
		NumTraits<float> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
	typedef math::Vector2<float> Real;
	typedef math::Vector2<float> NonInteger;
	typedef math::Vector2<float> Nested;
	enum {
		IsComplex = 0,
		IsInteger = 0,
		IsSigned = 1,
		RequireInitialization = 1,
		ReadCost = 1,
		AddCost = 2,
		MulCost = 6
	};
};

template<>
struct NumTraits<math::Vector2<double>>
:
		NumTraits<double> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
	typedef math::Vector2<double> Real;
	typedef math::Vector2<double> NonInteger;
	typedef math::Vector2<double> Nested;
	enum {
		IsComplex = 0,
		IsInteger = 0,
		IsSigned = 1,
		RequireInitialization = 1,
		ReadCost = 1,
		AddCost = 2,
		MulCost = 6
	};
};

template<>
struct NumTraits<math::Vector2<int>>
:
		NumTraits<int> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
	typedef math::Vector2<int> Real;
	typedef math::Vector2<int> Integer;
	typedef math::Vector2<int> Nested;
	enum {
		IsComplex = 0,
		IsInteger = 1,
		IsSigned = 1,
		RequireInitialization = 1,
		ReadCost = 1,
		AddCost = 2,
		MulCost = 6
	};
};

template<typename BinaryOp>
struct ScalarBinaryOpTraits<math::Vector2<double>, float, BinaryOp> {
	typedef math::Vector2<double> ReturnType;
};

template<typename BinaryOp>
struct ScalarBinaryOpTraits<float,math::Vector2<double>, BinaryOp> {
	typedef math::Vector2<double> ReturnType;
};

template<typename BinaryOp>
struct ScalarBinaryOpTraits<math::Vector2<double>, double, BinaryOp> {
	typedef math::Vector2<double> ReturnType;
};

template<typename BinaryOp>
struct ScalarBinaryOpTraits<double, math::Vector2<double>, BinaryOp> {
	typedef math::Vector2<double> ReturnType;
};


template<typename BinaryOp>
struct ScalarBinaryOpTraits<math::Vector2<float>, float, BinaryOp> {
	typedef math::Vector2<float> ReturnType;
};

template<typename BinaryOp>
struct ScalarBinaryOpTraits< float, math::Vector2<float>, BinaryOp> {
	typedef math::Vector2<float> ReturnType;
};

template<typename BinaryOp>
struct ScalarBinaryOpTraits<math::Vector2<float>, double, BinaryOp> {
	typedef math::Vector2<float> ReturnType;
};

template<typename BinaryOp>
struct ScalarBinaryOpTraits<double, math::Vector2<float>, BinaryOp> {
	typedef math::Vector2<float> ReturnType;
};

} // namespacd Eigen

