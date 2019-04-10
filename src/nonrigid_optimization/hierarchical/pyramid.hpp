/*
 * pyramid.hpp
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

//Standard library
#include <vector>
#include <memory>

//Libraries
#include <Eigen/Dense>

//local
#include "../../math/resampling.hpp"

namespace nonrigid_optimization {
namespace hierarchical{

/**
 * A pyramid representation of a discrete scalar field
 */
template <typename Container>
class Pyramid{
public:
	Pyramid(Container field, int maximum_chunk_size=8, math::DownsamplingStrategy downsampling_strategy
			= math::DownsamplingStrategy::AVERAGE);

	const Container& get_level(int i_level) const;
	size_t get_level_count() const;
private:
	std::vector<Container> levels;
};

} //namespace hierarchical
} //namespace nonrigid_optimization
