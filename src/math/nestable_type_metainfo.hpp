/*
 * nestable_type_metainfo.hpp
 *
 *  Created on: Mar 3, 2019
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

namespace Eigen {

// region ============================== VECTOR 2 ======================================================================

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
// endregion
// region ============================== VECTOR 3 ======================================================================
template<>
struct NumTraits<math::Vector3<float>>
:
		NumTraits<float> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
	typedef math::Vector3<float> Real;
	typedef math::Vector3<float> NonInteger;
	typedef math::Vector3<float> Nested;
	enum {
		IsComplex = 0,
		IsInteger = 0,
		IsSigned = 1,
		RequireInitialization = 1,
		ReadCost = 1,
		AddCost = 3,
		MulCost = 9
	};
};

template<>
struct NumTraits<math::Vector3<double>>
:
		NumTraits<double> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
	typedef math::Vector3<double> Real;
	typedef math::Vector3<double> NonInteger;
	typedef math::Vector3<double> Nested;
	enum {
		IsComplex = 0,
		IsInteger = 0,
		IsSigned = 1,
		RequireInitialization = 1,
		ReadCost = 1,
		AddCost = 3,
		MulCost = 9
	};
};

template<>
struct NumTraits<math::Vector3<int>>
:
		NumTraits<int> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
	typedef math::Vector3<int> Real;
	typedef math::Vector3<int> Integer;
	typedef math::Vector3<int> Nested;
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
struct ScalarBinaryOpTraits<math::Vector3<double>, float, BinaryOp> {
	typedef math::Vector3<double> ReturnType;
};

template<typename BinaryOp>
struct ScalarBinaryOpTraits<float,math::Vector3<double>, BinaryOp> {
	typedef math::Vector3<double> ReturnType;
};

template<typename BinaryOp>
struct ScalarBinaryOpTraits<math::Vector3<double>, double, BinaryOp> {
	typedef math::Vector3<double> ReturnType;
};

template<typename BinaryOp>
struct ScalarBinaryOpTraits<double, math::Vector3<double>, BinaryOp> {
	typedef math::Vector3<double> ReturnType;
};


template<typename BinaryOp>
struct ScalarBinaryOpTraits<math::Vector3<float>, float, BinaryOp> {
	typedef math::Vector3<float> ReturnType;
};

template<typename BinaryOp>
struct ScalarBinaryOpTraits< float, math::Vector3<float>, BinaryOp> {
	typedef math::Vector3<float> ReturnType;
};

template<typename BinaryOp>
struct ScalarBinaryOpTraits<math::Vector3<float>, double, BinaryOp> {
	typedef math::Vector3<float> ReturnType;
};

template<typename BinaryOp>
struct ScalarBinaryOpTraits<double, math::Vector3<float>, BinaryOp> {
	typedef math::Vector3<float> ReturnType;
};
// endregion
} // namespace Eigen



