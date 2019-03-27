/*
 * pyramid3d.hpp
 *
 *  Created on: Mar 1, 2019
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

//Libraries
#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>


namespace nonrigid_optimization {

/**
 * A pyramid representation of a continuous 3d field
 */
template<typename ElementType>
class Pyramid3d {
public:
	Pyramid3d(Eigen::Tensor<ElementType,3> field, int maximum_chunk_size=8);
	virtual ~Pyramid3d() = default;
	const Eigen::Tensor<ElementType,3>& level(int i_level) const;
	size_t level_count() const;
private:
	std::vector<Eigen::Tensor<ElementType,3>> levels;

};


}//namespace nonrigid_optimization


