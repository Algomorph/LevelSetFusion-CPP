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

//local
#include "pyramid3d.hpp"
#include "../math/checks.hpp"

namespace eig = Eigen;

namespace nonrigid_optimization {

template<typename ElementType>
Pyramid3d<ElementType>::Pyramid3d(eig::Tensor<ElementType, 3> field, int maximum_chunk_size) :
		levels() {

#ifndef NDEBUG
	eigen_assert((math::is_power_of_two(field.dimension(0))
			&& math::is_power_of_two(field.dimension(1))
			&& math::is_power_of_two(field.dimension(1)))
			&& "The argument 'field' must have a power of two for each dimension.");

	eigen_assert(math::is_power_of_two(maximum_chunk_size) && // @suppress("Invalid arguments")
			"The argument 'maximum_chunk_size' must be an integer power of 2, i.e. 4, 8, 16, etc.");
#endif
	int power_of_two_largest_chunk = static_cast<int>(std::log2(maximum_chunk_size));
#ifndef NDEBUG
	//check that we can get a level with the maximum chunk size
	int max_level_count = static_cast<int>(
			std::min( { std::log2(field.dimension(0)), std::log2(field.dimension(1)), std::log2(field.dimension(2)) }));
	eigen_assert(max_level_count < power_of_two_largest_chunk && "Maximum chunk size too large for the field size."); // @suppress("Invalid arguments")
#endif

	int level_count = power_of_two_largest_chunk + 1;
	levels.push_back(field);
	eig::Tensor<ElementType, 3>* previous_level = &field;

	for (int i_level = 1; i_level < level_count; i_level++) {
		eig::Tensor<ElementType, 3> current_level(
				previous_level->dimension(0) / 2,
				previous_level->dimension(1) / 2,
				previous_level->dimension(2) / 2
						);
		//average each square of 4 cells into one
		for (int i_current_level_z = 0, i_previous_level_z = 0;
				i_current_level_z < current_level.dimension(2);
				i_current_level_z++, i_previous_level_z += 2) {
			for (eig::Index i_current_level_y = 0, i_previous_level_y = 0;
					i_current_level_y < current_level.dimension(1);
					i_current_level_y++, i_previous_level_y += 2) {
				for (eig::Index i_current_level_x = 0, i_previous_level_x = 0;
						i_current_level_x < current_level.dimension(0);
						i_current_level_x++, i_previous_level_x += 2) {
					/* @formatter:off -- haha, have to wait for CDT 9.7 for @formatter:off to work, awesome */
					current_level(i_current_level_x, i_current_level_y, i_current_level_z) = (
							(*previous_level)(i_previous_level_x, i_previous_level_y, i_previous_level_z) +
									(*previous_level)(i_previous_level_x + 1, i_previous_level_y, i_previous_level_z) +
									(*previous_level)(i_previous_level_x, i_previous_level_y + 1, i_previous_level_z) +
									(*previous_level)(i_previous_level_x + 1, i_previous_level_y + 1, i_previous_level_z) +
									(*previous_level)(i_previous_level_x, i_previous_level_y, i_previous_level_z + 1) +
									(*previous_level)(i_previous_level_x + 1, i_previous_level_y, i_previous_level_z + 1) +
									(*previous_level)(i_previous_level_x, i_previous_level_y + 1, i_previous_level_z + 1)
									+
									(*previous_level)(i_previous_level_x + 1, i_previous_level_y + 1,
											i_previous_level_z + 1)
							) / 8.0f;
					/* @formatter:on */
				}
			}
		}
		levels.push_back(current_level);
		previous_level = &current_level;
	}

	//levels should be ordered from coarsest to finest, reverse the order
	std::reverse(levels.begin(), levels.end());
}

template<typename ElementType>
const eig::Tensor<ElementType, 3>& Pyramid3d<ElementType>::level(int i_level) const {
	return this->levels[i_level];
}

template<typename ElementType>
size_t Pyramid3d<ElementType>::level_count() const{
	return this->levels.size();
}

} //namespace nonrigid_optimization
