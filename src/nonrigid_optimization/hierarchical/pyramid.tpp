/*
 * pyramid.tpp
 *
 *  Created on: Apr 10, 2019
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
#include "pyramid.hpp"
#include "../../error_handling/throw_assert.hpp"
#include "../../math/checks.hpp"
#include "../../math/resampling.hpp"

namespace nonrigid_optimization {
namespace hierarchical {

template<typename Scalar>
static inline
int get_max_level_count(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& field){
	return static_cast<int>(std::min(std::log2(field.rows()), std::log2(field.cols()))) + 1;
}

template<typename Scalar>
static inline
int get_max_level_count(const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& field){
	return static_cast<int>( std::min(
			{ std::log2(field.dimension(0)), std::log2(field.dimension(1)), std::log2(field.dimension(2)) })) + 1;
}


/**
 * @param field -- scalar field
 * @param maximum_chunk_size -- size of the chunk of the original field represented by a single element at the coarsest level
 */
template <typename Container>
Pyramid<Container>::Pyramid(Container field, int maximum_chunk_size, math::DownsamplingStrategy downsampling_strategy)
: levels(){
		throw_assert(math::is_power_of_two(maximum_chunk_size),
				"The argument 'maximum_chunk_size' must be an integer power of 2, i.e. 4, 8, 16, etc.");

		int power_of_two_largest_chunk = static_cast<int>(std::log2(maximum_chunk_size));

		//check that we can get a level with the maximum chunk size
		int max_level_count = get_max_level_count(field);
		throw_assert(max_level_count > power_of_two_largest_chunk, "Maximum chunk size too large for the field size.");// @suppress("Invalid arguments")

		int level_count = power_of_two_largest_chunk + 1;
		levels.push_back(field);
		Container* previous_level = &field;

		for (int i_level = 1; i_level < level_count; i_level++) {
			Container current_level = math::downsampleX2(*previous_level, downsampling_strategy);
			levels.push_back(current_level);
			previous_level = &levels[levels.size()-1];
		}

		//levels should be ordered from coarsest to finest, reverse the order
		std::reverse(levels.begin(), levels.end());
}

template<typename Container>
const Container& Pyramid<Container>::get_level(int i_level) const {
	return this->levels[i_level];
}

template<typename Container>
size_t Pyramid<Container>::get_level_count() const {
	return this->levels.size();
}


} //namespace hierarchical
} //namespace nonrigid_optimization
