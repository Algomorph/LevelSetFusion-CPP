/*
 * convergence_report.hpp
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
//TODO: optimize imports
//stdlib
#include <iostream>

//libraries
#include <Eigen/Eigen>

//local
#include "../math/typedefs.hpp"
#include "../math/tensors.hpp"
#include "../telemetry/tsdf_difference_statistics.hpp"
#include "../telemetry/warp_delta_statistics.hpp"

namespace eig = Eigen;

namespace telemetry {
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

	bool operator==(const ConvergenceReport& rhs) {
		return this->iteration_count == rhs.iteration_count &&
				this->iteration_limit_reached == rhs.iteration_limit_reached &&
				this->warp_delta_statistics == rhs.warp_delta_statistics &&
				this->tsdf_difference_statistics == rhs.tsdf_difference_statistics;
	}
	bool operator!=(const ConvergenceReport& rhs) {
		return !(*this == rhs);
	}

};

std::ostream &operator<<(std::ostream &ostr, const ConvergenceReport &ts);
} //namespace telemetry
