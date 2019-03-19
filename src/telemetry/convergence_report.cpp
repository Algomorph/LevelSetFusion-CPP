/*
 * convergence_report.cpp
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

//local
#include "../telemetry/convergence_report.hpp"


namespace telementry {
ConvergenceReport::ConvergenceReport(int iteration_count,
		bool iteration_limit_reached,
		WarpDeltaStatistics warp_delta_statistics,
		TsdfDifferenceStatistics tsdf_difference_statistics)
:
		iteration_count(iteration_count),
				iteration_limit_reached(iteration_limit_reached),
				warp_delta_statistics(warp_delta_statistics),
				tsdf_difference_statistics(tsdf_difference_statistics) {
}

std::ostream &operator<<(std::ostream &ostr, const ConvergenceReport &ts){
	ostr << "===[convergence report]==="
			<< std::endl << "  iter count: " << ts.iteration_count
			<< std::endl << "  limit reached: " << ts.iteration_limit_reached
			<< std::endl << "--------------------------"
			<< std::endl << ts.warp_delta_statistics
			<< std::endl << "--------------------------"
			<< std::endl << ts.tsdf_difference_statistics
			<< std::endl << "==========================)"
			;
	return ostr;
}
} //namespace telemetry


