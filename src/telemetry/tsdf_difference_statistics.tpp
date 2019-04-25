/*
 * tsdf_difference_statistics.cpp
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

// local
#include "../telemetry/tsdf_difference_statistics.hpp"
#include "../math/filtered_statistics.hpp"
#include "../math/statistics.hpp"
#include "../math/almost_equal.hpp"
#include "../math/cwise_binary.hpp"
#include "../math/cwise_unary.hpp"

namespace telemetry {

template<typename Coordinates>
TsdfDifferenceStatistics<Coordinates>::TsdfDifferenceStatistics(
		float difference_min,
		float difference_max,
		float difference_mean,
		float difference_standard_deviation,
		Coordinates biggest_difference_location
		) :
		difference_min(difference_min),
				difference_max(difference_max),
				difference_mean(difference_mean),
				difference_standard_deviation(difference_standard_deviation),
				biggest_difference_location(biggest_difference_location)
{
}

template<typename Coordinates>
eig::VectorXf TsdfDifferenceStatistics<Coordinates>::to_array() {
	eig::VectorXf out(7);
	out << difference_min,
			difference_max,
			difference_mean,
			difference_standard_deviation,
			static_cast<float>(biggest_difference_location.x),
			static_cast<float>(biggest_difference_location.y)
	;
	return out;
}

template<typename Coordinates>
bool TsdfDifferenceStatistics<Coordinates>::operator==(const TsdfDifferenceStatistics<Coordinates>& rhs) {
	return math::almost_equal(this->difference_min, rhs.difference_min) &&
			math::almost_equal(this->difference_max, rhs.difference_max) &&
			math::almost_equal(this->difference_mean, rhs.difference_mean) &&
			math::almost_equal(this->difference_standard_deviation, rhs.difference_standard_deviation) &&
			this->biggest_difference_location == rhs.biggest_difference_location;
}

template<typename Coordinates>
bool TsdfDifferenceStatistics<Coordinates>::operator!=(const TsdfDifferenceStatistics<Coordinates>& rhs) {
	return !(*this == rhs);
}

template<typename Coordinates>
std::ostream &operator<<(std::ostream &ostr, const TsdfDifferenceStatistics<Coordinates> &ts){
	ostr << "[tsdf diff stats]"
			<< std::endl << "  min: " << ts.difference_min
			<< std::endl << "  max: " << ts.difference_max
			<< std::endl << "  mean: " << ts.difference_mean
			<< std::endl << "  std: " << ts.difference_standard_deviation
			<< std::endl << "  greatest diff at: (" << ts.biggest_difference_location << ")"
			;
	return ostr;
}

template<typename Coordinates, typename ScalarContainer>
TsdfDifferenceStatistics<Coordinates> build_tsdf_difference_statistics(const ScalarContainer& canonical_field,
		const ScalarContainer& live_field){
	ScalarContainer diff = math::cwise_abs(math::cwise_subtract(live_field,canonical_field)).eval();
	typename ScalarContainer::Scalar diff_min = math::minimum(diff);
	Coordinates max_location; float diff_max;
	math::locate_maximum(diff_max, max_location, diff);
	float diff_mean = math::mean(diff);
	float diff_std = math::std(diff);

	return TsdfDifferenceStatistics<Coordinates>(diff_min, diff_max, diff_mean, diff_std, max_location);
}

} //namespace telemetry
