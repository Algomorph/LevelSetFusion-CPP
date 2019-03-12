/*
 * logging.cpp
 *
 *  Created on: Nov 13, 2018
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

#include "logging.hpp"

namespace logging {
WarpDeltaStatistics::WarpDeltaStatistics(
		float ratio_above_min_threshold,
		float length_min,
		float length_max,
		float length_mean,
		float length_standard_deviation,
		math::Vector2i longest_warp_location,
		bool is_largest_below_min_threshold,
		bool is_largest_above_max_threshold
		)
:
		ratio_above_min_threshold(ratio_above_min_threshold),
				length_min(length_min),
				length_max(length_max),
				length_mean(length_mean),
				length_standard_deviation(length_standard_deviation),
				longest_warp_location(longest_warp_location),
				is_largest_below_min_threshold(is_largest_below_min_threshold),
				is_largest_above_max_threshold(is_largest_above_max_threshold)
{
}

eig::VectorXf WarpDeltaStatistics::to_array() {
	eig::VectorXf out(7);
	out << ratio_above_min_threshold,
			length_min,
			length_max,
			length_mean,
			length_standard_deviation,
			static_cast<float>(longest_warp_location.x),
			static_cast<float>(longest_warp_location.y),
			static_cast<float>(is_largest_below_min_threshold),
			static_cast<float>(is_largest_above_max_threshold)
	;
	return out;
}

TsdfDifferenceStatistics::TsdfDifferenceStatistics(
		float difference_min,
		float difference_max,
		float difference_mean,
		float difference_standard_deviation,
		math::Vector2i biggest_difference_location
		) :
		difference_min(difference_min),
				difference_max(difference_max),
				difference_mean(difference_mean),
				difference_standard_deviation(difference_standard_deviation),
				biggest_difference_location(biggest_difference_location)
{
}

eig::VectorXf TsdfDifferenceStatistics::to_array(){
	eig::VectorXf out(7);
	out <<  difference_min,
			difference_max,
			difference_mean,
			difference_standard_deviation,
			static_cast<float>(biggest_difference_location.x),
			static_cast<float>(biggest_difference_location.y)
	;
	return out;
}

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

} //namespace logging

