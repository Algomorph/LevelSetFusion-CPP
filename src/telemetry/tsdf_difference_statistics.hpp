/*
 * tsdf_difference_statistics.hpp
 *
 *  Created on: Mar 15, 2019
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
#include <iostream>

//libraries
#include <Eigen/Eigen>

//local
#include "../math/typedefs.hpp"
#include "../math/tensors.hpp"

namespace eig = Eigen;

namespace logging {
/**
 * A structure for logging statistics pertaining to numerical differences between corresponding locations in the
 * canonical and the live (target and source) TSDF fields after optimization
 */
struct TsdfDifferenceStatistics {
	float difference_min = 0.0f;
	float difference_max = 0.0f;
	float difference_mean = 0.0f;
	float difference_standard_deviation = 0.0f;
	math::Vector2i biggest_difference_location = math::Vector2i(0);

	TsdfDifferenceStatistics() = default;
	TsdfDifferenceStatistics(
			float difference_min,
			float difference_max,
			float difference_mean,
			float difference_standard_deviation,
			math::Vector2i biggest_difference_location
			);
	TsdfDifferenceStatistics(
			const eig::MatrixXf& canonical_field,
			const eig::MatrixXf& live_field
			);

	eig::VectorXf to_array();

	bool operator==(const TsdfDifferenceStatistics& rhs);
	bool operator!=(const TsdfDifferenceStatistics& rhs);
};

std::ostream &operator<<(std::ostream &ostr, const TsdfDifferenceStatistics &ts);
} //namespace logging
