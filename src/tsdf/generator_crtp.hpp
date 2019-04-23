/*
 * generator_crtp.hpp
 *
 *  Created on: Apr 23, 2019
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

//libraries
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

//local
#include "parameters.hpp"

namespace tsdf{

template<typename Generator, typename ScalarContainer>
class GeneratorCRTP {
public:
	GeneratorCRTP(const Parameters<ScalarContainer>& parameters);

	const Parameters<ScalarContainer> parameters;
	typedef typename ScalarContainer::Scalar ContainerScalar;

	//TODO: make this generate method protected & hidden+called by corresponding methods in 2D & 3D versions
	//(not overloads), since then we can make the parameters different and remove the "image_y_coordinate" parameter for
	//the 3D case.
	ScalarContainer generate(
			const Eigen::Matrix<unsigned short, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& depth_image,
			const Eigen::Matrix<ContainerScalar, 4, 4, Eigen::ColMajor>& camera_pose =
					Eigen::Matrix<ContainerScalar, 4, 4, Eigen::ColMajor>::Identity(),
			int image_y_coordinate = 0) const;
};

}  // namespace tsdf


