/*
 * pyramid2d.h
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
#include <vector>

//Libraries
#include <Eigen/Eigen>

#pragma once

namespace eig = Eigen;

namespace nonrigid_optimization {
namespace hierarchical{
inline static
bool is_power_of_two(int number){
	return !(number == 0) && !(number & (number - 1));
}

/**
 * A pyramid representation of a contiguous 2d scalar field
 */
class Pyramid2d {
public:
	Pyramid2d(eig::MatrixXf field, int maximum_chunk_size=8);
	virtual ~Pyramid2d();
	const eig::MatrixXf& get_level(int i_level) const;
	size_t get_level_count() const;
private:
	std::vector<eig::MatrixXf> levels;

};
} /* namespace hierarchical */
} /* namespace nonrigid_optimization */
