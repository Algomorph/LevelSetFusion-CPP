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

#include <Eigen/Eigen>

#include "../../src/math/stacking.hpp"
#include "../../src/math/typedefs.hpp"

namespace eig = Eigen;
namespace test_data {



static math::MatrixXv2f vector_field = [] {
	math::MatrixXv2f field(4, 4);
	field << math::Vector2f(0.66137378f, 0.22941163f), math::Vector2f(-0.79364663f, -0.51078996f),
			math::Vector2f(0.31330802f, -0.62231087f), math::Vector2f(0.38155258f, 0.25911068f),

			math::Vector2f(-0.93761754f, 0.22711085f), math::Vector2f(-0.84484027f, 0.74134703f),
			math::Vector2f(-0.77734907f, 0.31051154f), math::Vector2f(0.05594392f, 0.62550403f),

			math::Vector2f(0.24144975f, -0.03810476f), math::Vector2f(-0.83927967f, 0.2171229f),
			math::Vector2f(0.3517115f, -0.34761186f), math::Vector2f(-0.3781738f, 0.4708583f),

			math::Vector2f(-0.60896495f, 0.32025099f), math::Vector2f(0.11699246f, -0.98680021f),
			math::Vector2f(-0.96371592f, -0.93434108f), math::Vector2f(0.42603218f, 0.76691092f);
	return field;
}();

static math::MatrixXv2f vector_field2 = []{
		math::MatrixXv2f vector_field(4,4);
		vector_field <<
		math::Vector2f(0.8562016f,0.876527f),
		math::Vector2f(0.8056713f,0.31369442f),
		math::Vector2f(0.28571403f,0.38419583f),
		math::Vector2f(0.86377007f,0.9078812f),

		math::Vector2f(0.12255816f,0.22223428f),
		math::Vector2f(0.4487159f,0.7280231f),
		math::Vector2f(0.61369246f,0.43351218f),
		math::Vector2f(0.3545089f,0.33867624f),

		math::Vector2f(0.5658683f,0.53506494f),
		math::Vector2f(0.69546276f,0.9331944f),
		math::Vector2f(0.05706289f,0.06915309f),
		math::Vector2f(0.5286004f,0.9154799f),

		math::Vector2f(0.98797816f,0.60008055f),
		math::Vector2f(0.07343615f,0.10326899f),
		math::Vector2f(0.28764063f,0.05625961f),
		math::Vector2f(0.32258928f,0.84611595f);
		return vector_field;
}();



}//namespace test_data
