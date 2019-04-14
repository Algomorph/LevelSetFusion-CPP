//  ================================================================
//  Created by Gregory Kramida on 04/14/18.
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

//local
#include "stacking.hpp"
#include "../error_handling/throw_assert.hpp"

namespace math{

template<typename Scalar>
Eigen::Matrix<math::Vector2<Scalar>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
stack(const std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>& set){
	throw_assert(set.size() == 2, "Size of the set should be 2 to return Vector2 as the nested type");
	//TODO
	throw_assert(false, "Not Implemented");
}

template<typename Scalar>
Eigen::Tensor<Scalar, 3, Eigen::ColMajor>
stack(const std::vector<Eigen::Tensor<Scalar, Eigen::ColMajor>>& set){
	throw_assert(set.size() == 3, "Size of the set should be 2 to return Vector2 as the nested type");
	//TODO
	throw_assert(false, "Not Implemented");
}


} //namespace math
