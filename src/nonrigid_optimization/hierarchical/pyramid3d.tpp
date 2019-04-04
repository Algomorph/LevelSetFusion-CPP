/*
 * pyramid3d_2.hpp
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

//stdlib
#include <cassert>

//local
#include "pyramid3d.hpp"
#include "../../math/checks.hpp"
#include "../../math/resampling.tpp"


namespace eig = Eigen;

namespace nonrigid_optimization {

template<typename ElementType>
Pyramid3d<ElementType>::Pyramid3d(eig::Tensor<ElementType, 3> field, int maximum_chunk_size) :
		levels() {

	assert((math::is_power_of_two(field.dimension(0))
			&& math::is_power_of_two(field.dimension(1))
			&& math::is_power_of_two(field.dimension(1)))
			&& "The argument 'field' must have a power of two for each dimension.");

	assert(math::is_power_of_two(maximum_chunk_size) && // @suppress("Invalid arguments")
			"The argument 'maximum_chunk_size' must be an integer power of 2, i.e. 4, 8, 16, etc.");

	int power_of_two_largest_chunk = static_cast<int>(std::log2(maximum_chunk_size));

	//check that we can get a level with the maximum chunk size
	int max_level_count = static_cast<int>(
			std::min( { std::log2(field.dimension(0)), std::log2(field.dimension(1)), std::log2(field.dimension(2)) }))
			+ 1;
	assert(max_level_count > power_of_two_largest_chunk && "Maximum chunk size too large for the field size."); // @suppress("Invalid arguments")


	int level_count = power_of_two_largest_chunk + 1;
	levels.push_back(field);
	eig::Tensor<ElementType, 3>* previous_level = &field;

	for (int i_level = 1; i_level < level_count; i_level++) {
		levels.push_back(math::downsampleX2_average(*previous_level));
		previous_level = &levels[levels.size() - 1];
	}

	//levels should be ordered from coarsest to finest, reverse the order
	std::reverse(levels.begin(), levels.end());
}

template<typename ElementType>
const eig::Tensor<ElementType, 3>& Pyramid3d<ElementType>::level(int i_level) const {
	return this->levels[i_level];
}

template<typename ElementType>
size_t Pyramid3d<ElementType>::level_count() const {
	return this->levels.size();
}

} //namespace nonrigid_optimization
