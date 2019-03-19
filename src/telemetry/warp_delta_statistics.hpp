/*
 * warp_delta_statistics.hpp
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

namespace telementry {

/**
 * A structure for logging statistics pertaining to warps at the end of an optimization iteration
 * (in warp- or warp-update-threshold-based optimizations)
 */
struct WarpDeltaStatistics {

	float ratio_above_min_threshold = 0.0;
	float length_min = 0.0f;
	float length_max = 0.0f;
	float length_mean = 0.0;
	float length_standard_deviation = 0.0;
	math::Vector2i longest_warp_location = math::Vector2i(0);
	bool is_largest_below_min_threshold = false;
	bool is_largest_above_max_threshold = false;

	WarpDeltaStatistics() = default;
	WarpDeltaStatistics(
			float ratio_above_min_threshold,
			float length_min,
			float length_max,
			float length_mean,
			float length_standard_deviation,
			math::Vector2i longest_warp_location,
			bool is_largest_below_min_threshold,
			bool is_largest_above_max_threshold
			);
	WarpDeltaStatistics(const math::MatrixXv2f& warp_field,
			const eig::MatrixXf& canonical_field,
			const eig::MatrixXf& live_field,
			float min_threshold, float max_threshold);

	eig::VectorXf to_array();

	bool operator==(const WarpDeltaStatistics& rhs);
	bool operator!=(const WarpDeltaStatistics& rhs);
};

std::ostream &operator<<(std::ostream &ostr, const WarpDeltaStatistics &ts);

} //namespace telemetry
