/*
 * pyramid2d.cpp
 *
 *  Created on: Dec 18, 2018
 *      Author: Gregory Kramida
 *   Copyright: 2018 Gregory Kramida
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
//A pyramid representation of a contiguous 2d scalar field

//Standard library
#include <cmath>
#include <algorithm>
//Local
#include "pyramid2d.hpp"
#include "../../math/checks.hpp"
#include "../../math/resampling.hpp"

namespace nonrigid_optimization {
namespace hierarchical{

//TODO: Pyramid2d templated on the Scalar type to allow alternative element types
/**
 * @param field -- scalar field
 * @param maximum_chunk_size -- size of the chunk of the original field represented by a single element at the coarsest level
 */
Pyramid2d::Pyramid2d(eig::MatrixXf field, int maximum_chunk_size) :
		levels() {
#ifndef NDEBUG
	eigen_assert((math::is_power_of_two(field.rows()) && math::is_power_of_two(field.cols())) // @suppress("Invalid arguments")
			&& "The argument 'field' must have a power of two for each dimension.");
	eigen_assert(math::is_power_of_two(maximum_chunk_size) && // @suppress("Invalid arguments")
			"The argument 'maximum_chunk_size' must be an integer power of 2, i.e. 4, 8, 16, etc.");
#endif
	int power_of_two_largest_chunk = static_cast<int>(std::log2(maximum_chunk_size));
#ifndef NDEBUG
	//check that we can get a level with the maximum chunk size
	int max_level_count = static_cast<int>(std::min(std::log2(field.rows()), std::log2(field.cols()))) + 1;
	eigen_assert(max_level_count > power_of_two_largest_chunk && "Maximum chunk size too large for the field size."); // @suppress("Invalid arguments")
#endif

	int level_count = power_of_two_largest_chunk + 1;
	levels.push_back(field);
	eig::MatrixXf* previous_level = &field;

	for (int i_level = 1; i_level < level_count; i_level++) {
		eig::MatrixXf current_level = math::downsampleX2_nearest(*previous_level);
		levels.push_back(current_level);
		previous_level = &levels[levels.size()-1];
	}

	//levels should be ordered from coarsest to finest, reverse the order
	std::reverse(levels.begin(), levels.end());
}

Pyramid2d::~Pyramid2d()
{
}

const eig::MatrixXf& Pyramid2d::get_level(int i_level) const {
	return this->levels[i_level];
}

size_t Pyramid2d::get_level_count() const {
	return this->levels.size();
}
} /* namespace hierarchical */
} /* namespace nonrigid_optimization */
