/*
 *  logging.h
 *
 *  Created on: Nov 13, 2018
 *      Author: Gregory Kramida
 *  Copyright: 2018 Gregory Kramida
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
#include <Eigen/Eigen>

//local
#include "../math/typedefs.hpp"
#include "serializable.hpp"

namespace eig = Eigen;

namespace logging {

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

	eig::VectorXf to_array();
};

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

	eig::VectorXf to_array();
};

/**
 * A structure for logging characteristics of convergence (warp-threshold-based optimization) after a full optimization run
 */
struct ConvergenceReport {
	int iteration_count = 0;
	bool iteration_limit_reached = false;
	WarpDeltaStatistics warp_delta_statistics;
	TsdfDifferenceStatistics tsdf_difference_statistics;

	ConvergenceReport() = default;
	ConvergenceReport(
			int iteration_count,
			bool iteration_limit_reached,
			WarpDeltaStatistics warp_delta_statistics,
			TsdfDifferenceStatistics tsdf_difference_statistics
			);

};

} //namespace logging
