/*
 * warp_delta_statistics.cpp
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
#include "../telemetry/warp_delta_statistics.hpp"

#include "../math/filtered_statistics.hpp"
#include "../math/statistics.hpp"
#include "../math/almost_equal.hpp"


namespace telemetry {
template<typename Coordinates>
WarpDeltaStatistics<Coordinates>::WarpDeltaStatistics(
		float ratio_above_min_threshold,
		float length_min,
		float length_max,
		float length_mean,
		float length_standard_deviation,
		Coordinates longest_warp_location,
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


template<typename Coordinates>
eig::VectorXf WarpDeltaStatistics<Coordinates>::to_array() {
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

template<typename Coordinates>
bool WarpDeltaStatistics<Coordinates>::operator==(const WarpDeltaStatistics & rhs) {
	return math::almost_equal(this->ratio_above_min_threshold, rhs.ratio_above_min_threshold) &&
			math::almost_equal(this->length_min, rhs.length_min) &&
			math::almost_equal(this->length_max, rhs.length_max) &&
			math::almost_equal(this->length_mean, rhs.length_mean) &&
			math::almost_equal(this->length_standard_deviation, rhs.length_standard_deviation) &&
			this->longest_warp_location == rhs.longest_warp_location &&
			this->is_largest_below_min_threshold == rhs.is_largest_below_min_threshold &&
			this->is_largest_above_max_threshold == rhs.is_largest_above_max_threshold
	;
}

template<typename Coordinates>
bool WarpDeltaStatistics<Coordinates>::operator!=(const WarpDeltaStatistics & rhs) {
	return !(*this == rhs);
}

template<typename Coordinates, typename ScalarContainer, typename VectorContainer>
WarpDeltaStatistics<Coordinates> build_warp_delta_statistics(const VectorContainer& warp_field,
		const ScalarContainer& canonical_field,
		const ScalarContainer& live_field,
		float min_threshold, float max_threshold){
	float length_max;
	Coordinates longest_warp_location;
	math::locate_max_norm(length_max, longest_warp_location, warp_field);
	float length_mean, standard_deviation, ratio_above_min_threshold;
	math::mean_and_std_vector_length_band_union(length_mean, standard_deviation, warp_field, live_field,
			canonical_field);
	float length_min = math::min_norm(warp_field);
	ratio_above_min_threshold =
			math::ratio_of_vector_lengths_above_threshold_band_union(warp_field,
					min_threshold, live_field,
					canonical_field);
	WarpDeltaStatistics<Coordinates> result(
			ratio_above_min_threshold,
			length_min,
			length_max,
			length_mean,
			standard_deviation,
			longest_warp_location,
			length_max < min_threshold,
			length_max > max_threshold
			);
	return result;
}

template<typename Coordinates>
std::ostream& operator<<(std::ostream &ostr, const WarpDeltaStatistics<Coordinates> &ts)
		{
	ostr << "[warp delta stats]"
			<< std::endl << "  ratio: " << ts.ratio_above_min_threshold
			<< std::endl << "  min: " << ts.length_min
			<< std::endl << "  max: " << ts.length_max
			<< std::endl << "  mean: " << ts.length_mean
			<< std::endl << "  std: " << ts.length_standard_deviation
			<< std::endl << "  longest at: (" << ts.longest_warp_location << ")"
			<< std::endl << "  below min thresh: " << ts.is_largest_below_min_threshold
			<< std::endl << "  above max thresh: " << ts.is_largest_above_max_threshold
			;
	return ostr;
}

} //namespace telemetry
